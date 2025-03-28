from __future__ import annotations
from multiprocessing import context


def _pre_import():
    from ctypes import CDLL
    import os

    __file_dir__ = os.path.dirname(os.path.realpath(__file__))
    DNDSR_bin_dir = os.path.realpath(os.path.join(__file_dir__, "..", "bin"))
    DNDSR_libext_dir = os.path.realpath(
        os.path.join(__file_dir__, "..", "lib", "dndsr_external")
    )
    DNDSR_lib_dir = os.path.realpath(os.path.join(__file_dir__, "..", "lib"))
    if os.name == "posix":
        # controlled from CMake side to be compatible with them all
        # as we might be using a very new g++/clang++ with very new C++ runtime library
        # that could be incompatible with what libfmt.so finds first
        #! should do well without explicit import
        #! as fmt is static and libdnds.so is the first to require any libstdc++.so
        # CDLL(os.path.join(DNDSR_libext_dir, "libstdc++.so"))
        #! now statically linked:
        # CDLL(os.path.join(DNDSR_libext_dir, "libfmt.so"))
        CDLL(os.path.join(DNDSR_libext_dir, "libz.so"))
        CDLL(os.path.join(DNDSR_libext_dir, "libhdf5.so"))
        CDLL(os.path.join(DNDSR_libext_dir, "libcgns.so"))
        CDLL(os.path.join(DNDSR_libext_dir, "libmetis.so"))
        CDLL(os.path.join(DNDSR_libext_dir, "libparmetis.so"))

        # print(f"here {DNDSR_bin_dir}")
        # os.system(f"ls -la {DNDSR_bin_dir}")
        # name = os.path.join(DNDSR_bin_dir, "libdnds_shared.so")
        # os.system(f"ldd { name }")
        # while True:
        #     pass
        CDLL(os.path.join(DNDSR_lib_dir, "libdnds_shared.so"))
        # os.environ["LD_LIBRARY_PATH"] = (
        #     DNDSR_bin_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        # )
        pass
    elif os.name == "nt":
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
    print(f"module load: {__file__}")

# List the symbols you want to expose from core
# __all__ = dir(code)

# Expose symbols from core in the current module's namespace
# for symbol in __all__:
#     if symbol != "__name__":
#         globals()[symbol] = getattr(core, symbol)


import contextlib
import typing
import atexit
import sys


@contextlib.contextmanager
def MPI_Context(
    args: list[str] = [],
) -> typing.Generator[tuple[list[str], MPIInfo], None, None]:
    err, argsNew = MPI.Init_thread(args)
    if err:
        raise RuntimeError(f"MPI.Init_thread returned error code {err}")
    try:
        world = MPIInfo()
        world.setWorld()
        yield (argsNew, world)
    finally:
        MPI.Finalize()


def _init_mpi():
    #! we are ignoring any arguments here
    err, argsNew = MPI.Init_thread(sys.argv)
    if err:
        raise RuntimeError(f"MPI.Init_thread returned error code {err}")
    sys.argv = argsNew
    atexit.register(MPI.Finalize)


_init_mpi()


def _row_size_to_name(row_size: int | str) -> str:
    if isinstance(row_size, int):
        if row_size >= 0:
            return str(row_size)
        else:
            raise ValueError(f"row_size {row_size} below 0 is illegal")
    else:
        if str(row_size) == "D":
            return "D"
        elif str(row_size) in {"N", "I"}:
            return "I"
        elif row_size is None:
            return "None"
        else:
            raise ValueError(f"row_size {str(row_size)} is illegal")


def _array_value_type_to_name(type: str) -> str:
    if str(type) == "d":
        return "d"
    elif str(type) == "q":
        return "q"
    elif type is int:
        return "q"
    elif type is None:
        return "None"
    else:
        raise ValueError(f"unrecognized type for array {type}")


def _get_array_name(
    type: str = None,
    row_size: int | str | None = None,
    row_max: int | str | None = None,
    prepend: str = "Array",
    row_size_n: int | str | None = None,
    row_max_n: int | str | None = None,
) -> str:

    triedNames = []

    if (
        prepend == "Array"
        or prepend == "ParArray"
        or prepend == "ArrayTransformer"
        or prepend == "ParArrayPair"
    ):
        t_name = _array_value_type_to_name(type)
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        align_name = "D"
        className = f"{prepend}_{t_name}_{rs_name}_{rm_name}_{align_name}"
    elif prepend == "ArrayAdjacency" or prepend == "ArrayEigenVector":
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        align_name = "D"
        className = f"{prepend}_{rs_name}_{rm_name}_{align_name}"
    elif prepend == "ArrayEigenMatrix":
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        rs_name_n = _row_size_to_name(row_size_n)
        rm_name_n = rs_name_n if row_max_n is None else _row_size_to_name(row_max_n)
        align_name = "D"
        className = (
            f"{prepend}_{rs_name}x{rs_name_n}_{rm_name}x{rm_name_n}_{align_name}"
        )
    elif prepend == "ArrayEigenUniMatrixBatch":
        rs_name = _row_size_to_name(row_size)
        rs_name_n = _row_size_to_name(row_size_n)
        className = f"{prepend}_{rs_name}x{rs_name_n}"

    triedNames.append(className)

    # TODO: this decision queue does not guarantee coherence when the instantiation expands?
    # if className not in globals():
    #     className = f"{prepend}_{t_name}_{'D'}_{'D'}_{align_name}"
    #     triedNames.append(className)
    # if className not in globals() and rs_name == "N":
    #     className = f"{prepend}_{t_name}_{'N'}_{'D'}_{align_name}"
    #     triedNames.append(className)
    if className not in globals():
        raise ValueError(f"cannot find type, tried{triedNames}")

    return className


def Array(
    type: str, row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> Array_d_I_I_D:
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="Array")]
    return cls(*init_args)


def ParArray(
    type: str, row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ParArray_d_I_I_D:
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ParArray")]
    return cls(*init_args)


def ParArrayPair(
    type: str, row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ParArrayPair_d_I_I_D:
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ParArrayPair")]
    return cls(*init_args)


def ArrayTransformer(
    type: str, row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ArrayTransformer_d_3_3_D:
    cls = globals()[
        _get_array_name(type, row_size, row_max, prepend="ArrayTransformer")
    ]
    return cls(*init_args)


def ArrayAdjacency(
    row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ArrayAdjacency_I_I_D:
    cls = globals()[
        _get_array_name(row_size=row_size, row_max=row_max, prepend="ArrayAdjacency")
    ]
    return cls(*init_args)


def ArrayEigenMatrix(
    row_size: int | str,
    row_size_n: int | str,
    row_max: int | str = None,
    row_max_n: int | str = None,
    init_args: tuple = (),
) -> ArrayEigenMatrix_Dx1_Dx1_D:
    cls = globals()[
        _get_array_name(
            row_size=row_size,
            row_max=row_max,
            row_size_n=row_size_n,
            row_max_n=row_max_n,
            prepend="ArrayEigenMatrix",
        )
    ]
    return cls(*init_args)


def ArrayEigenUniMatrixBatch(
    row_size: int | str,
    row_size_n: int | str,
    init_args: tuple = (),
) -> ArrayEigenUniMatrixBatch_DxD:
    cls = globals()[
        _get_array_name(
            row_size=row_size,
            row_size_n=row_size_n,
            prepend="ArrayEigenUniMatrixBatch",
        )
    ]
    return cls(*init_args)


def ArrayEigenVector(
    row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ArrayEigenVector_I_I_D:
    cls = globals()[
        _get_array_name(row_size=row_size, row_max=row_max, prepend="ArrayEigenVector")
    ]
    return cls(*init_args)


def ArrayTransformerFromParArray(
    arr: ParArray_d_1_1_D, init_args: tuple = ()
) -> ArrayTransformer_d_1_1_D:
    cls = arr.getTrans()
    return cls(*init_args)


__all__ = ["MPI", "Debug"]

if __name__ == "__main__":
    print(__all__)
