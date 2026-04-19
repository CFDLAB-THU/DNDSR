"""
Correctness tests for CFV.FiniteVolume through the Python bindings.

Tests the FiniteVolume construction pipeline and verifies geometric
quantities (volumes, areas, barycenters, normals) against known values
for the Uniform_3x3_wall mesh (a 3x3 quad grid on [-1,2]^2).

These tests use pytest assertions and are intended for automated CI.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import create_mesh_from_CGNS, create_bnd_mesh


@pytest.fixture(scope="module")
def mpi():
    m = DNDS.MPIInfo()
    m.setWorld()
    return m


def _data_mesh_path(name: str) -> str:
    """Return absolute path to a mesh file under data/mesh/."""
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", name
    )


def _build_fv(mpi, mesh) -> CFV.FiniteVolume:
    """Construct a FiniteVolume with default settings on a given mesh."""
    fv = CFV.FiniteVolume(mpi, mesh)
    settings = fv.GetSettings()
    settings["intOrder"] = 3
    settings["maxOrder"] = 1
    fv.ParseSettings(settings)

    fv.SetCellAtrBasic()
    fv.ConstructCellVolume()
    fv.ConstructCellBary()
    fv.ConstructCellCent()
    fv.ConstructCellIntJacobiDet()
    fv.ConstructCellIntPPhysics()
    fv.ConstructCellAlignedHBox()
    fv.ConstructCellMajorHBoxCoordInertia()

    fv.SetFaceAtrBasic()
    fv.ConstructFaceCent()
    fv.ConstructFaceArea()
    fv.ConstructFaceIntJacobiDet()
    fv.ConstructFaceIntPPhysics()
    fv.ConstructFaceUnitNorm()
    fv.ConstructFaceMeanNorm()

    fv.ConstructCellSmoothScale()
    return fv


@pytest.fixture(scope="module")
def wall_mesh_fv(mpi):
    """Build FiniteVolume on the Uniform_3x3_wall mesh (9 quads, [-1,2]^2)."""
    mesh_file = _data_mesh_path("Uniform_3x3_wall.cgns")
    mesh, reader, name2Id = create_mesh_from_CGNS(mesh_file, mpi, 2)
    fv = _build_fv(mpi, mesh)
    return mesh, fv, name2Id


@pytest.fixture(scope="module")
def periodic_mesh_fv(mpi):
    """Build FiniteVolume on Uniform_3x3 with periodic BCs (9 quads, [0,3]^2)."""
    mesh_file = _data_mesh_path("Uniform_3x3.cgns")
    mesh, reader, name2Id = create_mesh_from_CGNS(
        mesh_file,
        mpi,
        2,
        periodic_geometry={
            "translation1": [3, 0, 0],
            "translation2": [0, 3, 0],
        },
    )
    fv = _build_fv(mpi, mesh)
    return mesh, fv, name2Id


# ===================================================================
# Cell volume tests
# ===================================================================


class TestCellVolumes:
    """Verify cell volumes for uniform quad meshes."""

    def test_wall_mesh_cell_volumes(self, wall_mesh_fv):
        """Each cell in Uniform_3x3_wall has volume 1.0 (unit squares on [-1,2]^2)."""
        mesh, fv, _ = wall_mesh_fv
        for iCell in range(mesh.NumCell()):
            vol = fv.GetCellVol(iCell)
            assert vol == pytest.approx(1.0, abs=1e-12), (
                f"Cell {iCell} volume = {vol}, expected 1.0"
            )

    def test_periodic_mesh_cell_volumes(self, periodic_mesh_fv):
        """Each cell in Uniform_3x3 has volume 1.0 (unit squares on [0,3]^2)."""
        mesh, fv, _ = periodic_mesh_fv
        for iCell in range(mesh.NumCell()):
            vol = fv.GetCellVol(iCell)
            assert vol == pytest.approx(1.0, abs=1e-12), (
                f"Cell {iCell} volume = {vol}, expected 1.0"
            )


# ===================================================================
# Global volume tests
# ===================================================================


class TestGlobalVolume:
    """Verify global (summed) volume equals the domain area."""

    def test_wall_mesh_global_volume(self, wall_mesh_fv):
        """Domain [-1,2]^2 has area 9.0."""
        _, fv, _ = wall_mesh_fv
        assert fv.GetGlobalVol() == pytest.approx(9.0, abs=1e-10)

    def test_periodic_mesh_global_volume(self, periodic_mesh_fv):
        """Domain [0,3]^2 has area 9.0."""
        _, fv, _ = periodic_mesh_fv
        assert fv.GetGlobalVol() == pytest.approx(9.0, abs=1e-10)


# ===================================================================
# Face area tests
# ===================================================================


class TestFaceAreas:
    """Verify face areas on uniform quad meshes are 1.0 (unit edge length)."""

    def test_wall_mesh_face_areas(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        for iFace in range(mesh.NumFaceProc()):
            area = fv.GetFaceArea(iFace)
            assert area == pytest.approx(1.0, abs=1e-12), (
                f"Face {iFace} area = {area}, expected 1.0"
            )

    def test_periodic_mesh_face_areas(self, periodic_mesh_fv):
        mesh, fv, _ = periodic_mesh_fv
        for iFace in range(mesh.NumFaceProc()):
            area = fv.GetFaceArea(iFace)
            assert area == pytest.approx(1.0, abs=1e-12), (
                f"Face {iFace} area = {area}, expected 1.0"
            )


# ===================================================================
# Cell barycenter tests
# ===================================================================


class TestCellBarycenters:
    """Verify cell barycenters are at cell centers."""

    def test_wall_mesh_barycenters_in_domain(self, wall_mesh_fv):
        """All barycenters should be within [-1,2]^2."""
        mesh, fv, _ = wall_mesh_fv
        for iCell in range(mesh.NumCell()):
            bary = fv.GetCellBary(iCell)
            assert -1.0 - 1e-10 <= bary[0] <= 2.0 + 1e-10, (
                f"Cell {iCell} bary x = {bary[0]} out of range"
            )
            assert -1.0 - 1e-10 <= bary[1] <= 2.0 + 1e-10, (
                f"Cell {iCell} bary y = {bary[1]} out of range"
            )

    def test_wall_mesh_barycenters_are_half_integers(self, wall_mesh_fv):
        """Barycenters of unit-square cells are at half-integer offsets from -1."""
        mesh, fv, _ = wall_mesh_fv
        expected_coords = {-0.5, 0.5, 1.5}
        bary_xs = set()
        bary_ys = set()
        for iCell in range(mesh.NumCell()):
            bary = fv.GetCellBary(iCell)
            bary_xs.add(round(bary[0], 8))
            bary_ys.add(round(bary[1], 8))
        assert bary_xs == expected_coords, f"Got x coords: {bary_xs}"
        assert bary_ys == expected_coords, f"Got y coords: {bary_ys}"


# ===================================================================
# DOF array building
# ===================================================================


class TestDOFArrays:
    """Verify DOF arrays can be built and have correct sizes."""

    def test_build_udof_dynamic(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        u = CFV.tUDof_D()
        fv.BuildUDof_D(u, 1)
        assert u.Size() >= mesh.NumCell()
        assert u.father.Size() == mesh.NumCell()

    def test_build_udof_fixed_5(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        u = CFV.tUDof_5()
        fv.BuildUDof_5(u, 5)
        assert u.Size() >= mesh.NumCell()

    def test_build_ugrad_3xD(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        grad = CFV.tUGrad_3xD()
        fv.BuildUGrad_3xD(grad, 1)
        assert grad.Size() >= mesh.NumCell()

    def test_udof_write_read(self, wall_mesh_fv):
        """Writing and reading back DOF values through numpy views."""
        mesh, fv, _ = wall_mesh_fv
        u = CFV.tUDof_D()
        fv.BuildUDof_D(u, 1)
        for iCell in range(mesh.NumCell()):
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = float(iCell) + 0.5
        for iCell in range(mesh.NumCell()):
            u_i = np.array(u[iCell], copy=False)
            assert u_i[0] == pytest.approx(float(iCell) + 0.5, abs=1e-14)


# ===================================================================
# Array bytes reporting
# ===================================================================


class TestArrayBytes:
    """Verify getArrayBytes returns a positive value."""

    def test_array_bytes_positive(self, wall_mesh_fv):
        _, fv, _ = wall_mesh_fv
        nbytes = fv.getArrayBytes()
        assert nbytes > 0


# ===================================================================
# Smooth scale ratio
# ===================================================================


class TestSmoothScale:
    """Verify smooth scale ratio is finite and positive for all cells."""

    def test_smooth_scale_ratio(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        for iCell in range(mesh.NumCell()):
            ratio = fv.GetCellSmoothScaleRatio(iCell)
            assert np.isfinite(ratio), f"Cell {iCell} smooth scale ratio not finite"
            assert ratio > 0, f"Cell {iCell} smooth scale ratio = {ratio}"


# ===================================================================
# Jacobi determinant at quadrature points
# ===================================================================


class TestJacobiDet:
    """Verify Jacobian determinants at quadrature points are positive."""

    def test_cell_jacobi_det_positive(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        for iCell in range(mesh.NumCell()):
            # iG=0 is the first quadrature point
            jdet = fv.GetCellJacobiDet(iCell, 0)
            assert jdet > 0, f"Cell {iCell} Jacobi det = {jdet}"

    def test_face_jacobi_det_positive(self, wall_mesh_fv):
        mesh, fv, _ = wall_mesh_fv
        for iFace in range(mesh.NumFaceProc()):
            jdet = fv.GetFaceJacobiDet(iFace, 0)
            assert jdet > 0, f"Face {iFace} Jacobi det = {jdet}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
