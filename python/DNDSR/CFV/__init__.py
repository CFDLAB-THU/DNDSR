"""DNDSR.CFV — Compact Finite Volume, variational reconstruction."""

from __future__ import annotations

from DNDSR._loader import preload

preload("cfv")

from ._ext.cfv_pybind11 import *  # noqa: F401,F403

__all__: list[str] = []
