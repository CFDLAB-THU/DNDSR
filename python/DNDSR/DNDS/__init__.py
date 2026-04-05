"""DNDSR.DNDS — Core data structures, MPI arrays, serialization.

This module preloads the required shared libraries, imports the C++ pybind11
bindings, initializes MPI, and provides factory functions for Array types.
"""

from __future__ import annotations

import contextlib
import typing
import atexit
import sys

# --- Preload native shared libraries ---
from DNDSR._loader import preload

preload("dnds")

# --- Import C++ bindings ---
from ._ext.dnds_pybind11 import *  # noqa: F401,F403
from ._ext.dnds_pybind11 import MPI  # noqa: F401
from ._ext.dnds_pybind11 import Debug  # noqa: F401


# --- MPI initialization ---

@contextlib.contextmanager
def MPI_Context(
    args: list[str] = [],
) -> typing.Generator[tuple[list[str], MPIInfo], None, None]:  # type: ignore[name-defined]
    """Context manager that calls MPI.Init_thread / MPI.Finalize."""
    err, argsNew = MPI.Init_thread(args)
    if err:
        raise RuntimeError(f"MPI.Init_thread returned error code {err}")
    try:
        world = MPIInfo()  # type: ignore[name-defined]
        world.setWorld()
        yield (argsNew, world)
    finally:
        MPI.Finalize()


def _init_mpi() -> None:
    err, argsNew = MPI.Init_thread(sys.argv)
    if err:
        raise RuntimeError(f"MPI.Init_thread returned error code {err}")
    sys.argv = argsNew
    atexit.register(MPI.Finalize)


_init_mpi()


# --- Array factory helpers ---

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

    if prepend in {"Array", "ParArray", "ArrayTransformer", "ParArrayPair"}:
        t_name = _array_value_type_to_name(type)
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        align_name = "N"
        className = f"{prepend}_{t_name}_{rs_name}_{rm_name}_{align_name}"
    elif prepend in {
        "ArrayAdjacency", "ArrayAdjacencyPair",
        "ArrayEigenVector", "ArrayEigenVectorPair",
    }:
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        align_name = "N"
        className = f"{prepend}_{rs_name}_{rm_name}_{align_name}"
    elif prepend in {"ArrayEigenMatrix", "ArrayEigenMatrixPair"}:
        rs_name = _row_size_to_name(row_size)
        rm_name = rs_name if row_max is None else _row_size_to_name(row_max)
        rs_name_n = _row_size_to_name(row_size_n)
        rm_name_n = rs_name_n if row_max_n is None else _row_size_to_name(row_max_n)
        align_name = "N"
        className = (
            f"{prepend}_{rs_name}x{rs_name_n}_{rm_name}x{rm_name_n}_{align_name}"
        )
    elif prepend in {"ArrayEigenUniMatrixBatch", "ArrayEigenUniMatrixBatchPair"}:
        rs_name = _row_size_to_name(row_size)
        rs_name_n = _row_size_to_name(row_size_n)
        className = f"{prepend}_{rs_name}x{rs_name_n}"
    else:
        raise ValueError(f"prepend {prepend} not supported")

    triedNames.append(className)
    if className not in globals():
        raise ValueError(f"cannot find type, tried {triedNames}")
    return className


def Array(type: str, row_size: int | str, row_max: int | str = None,
          init_args: tuple = ()):
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="Array")]
    return cls(*init_args)


def ParArray(type: str, row_size: int | str, row_max: int | str = None,
             init_args: tuple = ()):
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ParArray")]
    return cls(*init_args)


def ParArrayPair(type: str, row_size: int | str, row_max: int | str = None,
                 init_args: tuple = ()):
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ParArrayPair")]
    return cls(*init_args)


def ArrayTransformer(type: str, row_size: int | str, row_max: int | str = None,
                     init_args: tuple = ()):
    cls = globals()[_get_array_name(type, row_size, row_max, prepend="ArrayTransformer")]
    return cls(*init_args)


def ArrayAdjacency(row_size: int | str, row_max: int | str = None,
                   init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    prepend="ArrayAdjacency")]
    return cls(*init_args)


def ArrayAdjacencyPair(row_size: int | str, row_max: int | str = None,
                       init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    prepend="ArrayAdjacencyPair")]
    return cls(*init_args)


def ArrayEigenMatrix(row_size: int | str, row_size_n: int | str,
                     row_max: int | str = None, row_max_n: int | str = None,
                     init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    row_size_n=row_size_n, row_max_n=row_max_n,
                                    prepend="ArrayEigenMatrix")]
    return cls(*init_args)


def ArrayEigenMatrixPair(row_size: int | str, row_size_n: int | str,
                         row_max: int | str = None, row_max_n: int | str = None,
                         init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    row_size_n=row_size_n, row_max_n=row_max_n,
                                    prepend="ArrayEigenMatrixPair")]
    return cls(*init_args)


def ArrayEigenUniMatrixBatch(row_size: int | str, row_size_n: int | str,
                             init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_size_n=row_size_n,
                                    prepend="ArrayEigenUniMatrixBatch")]
    return cls(*init_args)


def ArrayEigenUniMatrixBatchPair(row_size: int | str, row_size_n: int | str,
                                 init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_size_n=row_size_n,
                                    prepend="ArrayEigenUniMatrixBatchPair")]
    return cls(*init_args)


def ArrayEigenVector(row_size: int | str, row_max: int | str = None,
                     init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    prepend="ArrayEigenVector")]
    return cls(*init_args)


def ArrayEigenVectorPair(row_size: int | str, row_max: int | str = None,
                         init_args: tuple = ()):
    cls = globals()[_get_array_name(row_size=row_size, row_max=row_max,
                                    prepend="ArrayEigenVectorPair")]
    return cls(*init_args)


def ArrayTransformerFromParArray(arr, init_args: tuple = ()):
    cls = arr.getTrans()
    return cls(*init_args)


__all__ = ["MPI", "Debug"]
