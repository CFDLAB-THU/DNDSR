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


# Select one representative parameterization in each smoke lane. Other variants
# of these functions still run in routine testing, including at two MPI ranks.
_SMOKE_LANES = {
    "test/DNDS/test_view_ownership.py::test_flat_view_retains_allocation_not_array": "serial",
    "test/DNDS/test_runtime_array_bridge.py::test_one_pod_schema_spans_native_python_mpi_and_direct_io[True-h5]": "mpi",
    "test/CFV/test_fv_facility.py::test_reconstruction_shared_buffers_explicit_halo_and_point_sampling[True]": "mpi",
}
_SMOKE_FILES = {name.split("::")[0] for name in _SMOKE_LANES}


def pytest_addoption(parser):
    parser.addoption("--dnds-test-level", type=int, choices=(0, 1, 2), default=2,
                     help="Highest DNDSR test tier to collect (default: full qualification)")
    parser.addoption("--dnds-ci-lane", choices=("serial", "mpi"),
                     help="Select the CI single-process or distributed catalog")


# Distributed CI runs all Runtime/solver orchestration cases and selected mesh
# and CFV fixtures. Empty prefixes select every case in that module.
_MPI_MODULES = {
    "test/demos/test_runtime_solver.py": ("",),
    "test/Cauchy/test_operators.py": ("",),
    "test/Geom/test_native_mesh_helpers.py": ("test_default_native_archive_file_faults_reject_collectively_and_retry",),
    "test/CFV/test_fv_facility.py": ("test_reconstruction_shared_buffers_explicit_halo_and_point_sampling",),
}


def _mpi_prefixes(path):
    if path.startswith("test/DNDS/test_runtime_"):
        return ("",)
    return _MPI_MODULES.get(path, ())


def pytest_ignore_collect(collection_path, config):
    if collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        path = collection_path.relative_to(config.rootpath).as_posix()
        if config.getoption("--dnds-test-level") == 0:
            return path not in _SMOKE_FILES
        if config.getoption("--dnds-ci-lane") == "mpi":
            return not _mpi_prefixes(path)
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    maximum = config.getoption("--dnds-test-level")
    lane = config.getoption("--dnds-ci-lane")
    selected, deselected = [], []
    for item in items:
        smoke_lane = _SMOKE_LANES.get(item.nodeid)
        level = 1
        if smoke_lane:
            level = 0
        elif item.get_closest_marker("level2"):
            level = 2
        item.add_marker(f"level{level}")
        if smoke_lane:
            eligible = lane is None or lane == smoke_lane
        else:
            path, _, case = item.nodeid.partition("::")
            eligible = lane != "mpi" or case.startswith(_mpi_prefixes(path))
        (selected if level <= maximum and eligible else deselected).append(item)
    # A renamed smoke case must fail the CI lane even if its other tests pass.
    if maximum == 0 and lane and any(Path(arg).resolve() == Path(__file__).parent for arg in config.args):
        expected = {node for node, owner in _SMOKE_LANES.items() if owner == lane}
        missing = expected.difference(item.nodeid for item in selected)
        if missing:
            raise pytest.UsageError("Missing level-0 cases: " + ", ".join(sorted(missing)))
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
