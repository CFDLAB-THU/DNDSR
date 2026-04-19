"""Root conftest — ensures the in-tree ``python/`` directory is importable.

With this file ``pytest test/`` (or ``python -m pytest test/``) works
without ``pip install -e .`` as long as the C++ shared libraries and
pybind11 ``.so`` modules have been built and installed into the
``python/DNDSR/`` tree via::

    cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
    cmake --install build --component py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PYTHON_DIR = str(Path(__file__).resolve().parent.parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from DNDSR import DNDS  # noqa: E402


@pytest.fixture
def mpi():
    """Shared MPI fixture — creates a world-scope MPIInfo."""
    world = DNDS.MPIInfo()
    world.setWorld()
    yield world
