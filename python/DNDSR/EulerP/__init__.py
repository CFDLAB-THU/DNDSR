"""DNDSR.EulerP — Compressible Navier-Stokes evaluator with optional CUDA."""

from __future__ import annotations

from DNDSR._loader import preload

preload("eulerP")

from ._ext.eulerP_pybind11 import *  # noqa: F401,F403

__all__: list[str] = []
