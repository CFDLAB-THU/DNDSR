"""Small headless geometry oracles for the native mesh setup demos."""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI


pytestmark = pytest.mark.level2


def _demo(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source():
    return str(Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns")


def _quad(vertices):
    vertices = np.asarray(vertices)
    assert vertices.shape == (4, 2)
    # The source is the nine unit squares on [-1, 2] x [-1, 2].
    np.testing.assert_allclose(vertices, np.rint(vertices), atol=1e-12)
    assert np.min(vertices) >= -1 - 1e-12
    assert np.max(vertices) <= 2 + 1e-12
    area = 0.5 * abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))
                     - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    assert area == pytest.approx(1.0, abs=1e-12)
    return tuple(np.round(vertices.mean(axis=0), 12))


def _all_owned_centers(local):
    centers = [center for part in MPI.COMM_WORLD.allgather(local) for center in part]
    assert sorted(centers) == list(product((-0.5, 0.5, 1.5), repeat=2))


@pytest.mark.mpi
def test_cart_demo_native_setup_has_exact_geometry():
    mesh, reader = _demo("cart_overset_test_2D").get_mesh_2D(_source())
    assert mesh.has_native_state
    assert reader.mesh is mesh
    assert mesh.NumCellGlobal() == 9
    assert mesh.NumNodeGlobal() == 16
    assert mesh.NumFaceGlobal() == 24
    assert mesh.NumBndGlobal() == 12
    centers = []
    with mesh.acquire_read(faces=True) as read:
        owned, _, combined = read.counts["cell"]
        for cell in range(combined):
            points = read.cell_coords(cell)
            assert points.shape == (4, 3)
            np.testing.assert_allclose(points[:, 2], 0, atol=1e-12)
            center = _quad(points[:, :2])
            if cell < owned:
                centers.append(center)
                faces = read.cell_faces(cell)
                assert len(faces) == 4
                assert np.all((faces >= 0) & (faces < mesh.NumFaceProc()))
    _all_owned_centers(centers)


@pytest.mark.mpi
def test_draw_demo_retains_owned_and_ghost_polygons(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    path = tmp_path / f"demo-{MPI.COMM_WORLD.rank}.png"
    fig = _demo("draw_meshes_2D").draw_meshes_2D(_source(), str(path))
    try:
        assert path.is_file() and path.stat().st_size > 0
        assert fig.axes[0].get_title() == f"part_{MPI.COMM_WORLD.rank}"
        owned = []
        ghosts = 0
        for polygon in fig.axes[0].patches:
            vertices = polygon.get_xy()
            np.testing.assert_allclose(vertices[0], vertices[-1])
            center = _quad(vertices[:-1])
            color = polygon.get_facecolor()
            assert color in (to_rgba("blue"), to_rgba("red"))
            if color == to_rgba("blue"):
                owned.append(center)
            else:
                ghosts += 1
        _all_owned_centers(owned)
        total_ghosts = MPI.COMM_WORLD.allreduce(ghosts)
        if MPI.COMM_WORLD.size > 1:
            assert total_ghosts > 0
        else:
            assert total_ghosts == 0
    finally:
        plt.close(fig)
