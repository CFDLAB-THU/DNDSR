"""DNDSR.Geom — Unstructured mesh, CGNS I/O, partitioning."""

from __future__ import annotations

from DNDSR._loader import preload

preload("geom")

from ._ext.geom_pybind11 import *  # noqa: F401,F403
from ._ext.geom_pybind11 import Elem  # noqa: F401

__all__ = ["Elem"]
