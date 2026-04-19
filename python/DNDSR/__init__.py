"""DNDSR — Compact Finite Volume CFD framework.

Submodules:
    DNDS   — Core data structures, MPI arrays, serialization.
    Geom   — Unstructured mesh, CGNS I/O, partitioning.
    CFV    — Compact Finite Volume, variational reconstruction.
    EulerP — Compressible Navier–Stokes evaluator (with optional CUDA).
"""

from __future__ import annotations

from . import DNDS
from . import Geom
from . import CFV
from . import EulerP

__all__ = ["DNDS", "Geom", "CFV", "EulerP"]
