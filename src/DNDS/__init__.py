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

# List the symbols you want to expose from core
# __all__ = dir(code)

# Expose symbols from core in the current module's namespace
# for symbol in __all__:
#     if symbol != "__name__":
#         globals()[symbol] = getattr(core, symbol)


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
        else:
            raise ValueError(f"row_size {str(row_size)} is illegal")


def _array_value_type_to_name(type: str) -> str:
    if str(type) == "d":
        return "d"
    elif str(type) == "q":
        return "q"
    elif type is int:
        return "q"
    else:
        raise ValueError(f"unrecognized type for array {type}")


def _get_array_name(
    type: str, row_size: int | str, row_max: int | str = None, prepend: str = "Array"
) -> str:
    t_name = _array_value_type_to_name(type)
    rs_name = _row_size_to_name(row_size)
    rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
    align_name = "D"

    triedNames = []

    className = f"{prepend}_{t_name}_{rs_name}_{rm_name}_{align_name}"
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
) -> Array_d_3_3_D:
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="Array")]
    return cls(*init_args)


def ParArray(
    type: str, row_size: int | str, row_max: int | str = None, init_args: tuple = ()
) -> ParArray_d_3_3_D:
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ParArray")]
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
) -> ArrayAdjacency_q_I_I_D:
    cls = globals()[_get_array_name("q", row_size, row_max, prepend="ArrayAdjacency")]
    return cls(*init_args)


__all__ = ["MPI", "Debug"]

if __name__ == "__main__":
    print(__all__)
