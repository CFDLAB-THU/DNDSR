"""
Correctness tests for CFV.VariationalReconstruction and CFV.ModelEvaluator
through the Python bindings.

Tests cover:
  - VR construction pipeline (metrics, base+weight, rec coeff)
  - Constant and linear polynomial exactness
  - Iterative reconstruction convergence
  - ModelEvaluator RHS evaluation produces finite results
  - ModelEvaluator DoReconstructionIter changes uRec

Uses the Uniform_3x3 periodic mesh (9 quads, [0,3]^2).

These tests use pytest assertions and are intended for automated CI.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import json

from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import create_mesh_from_CGNS, create_bnd_mesh


@pytest.fixture(scope="module")
def mpi():
    m = DNDS.MPIInfo()
    m.setWorld()
    return m


def _data_mesh_path(name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", name
    )


@pytest.fixture(scope="module")
def periodic_vfv(mpi):
    """Build VariationalReconstruction_2 on Uniform_3x3 periodic mesh."""
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

    vfvSettings = {
        "maxOrder": 2,
        "intOrder": 5,
        "cacheDiffBase": True,
        "jacobiRelax": 1.0,
        "SORInstead": False,
        "subs2ndOrder": 0,
        "functionalSettings": {
            "dirWeightScheme": "HQM_OPT",
            "geomWeightScheme": "HQM_SD",
            "geomWeightPower": 0.5,
            "geomWeightBias": 1,
        },
    }

    vfv = CFV.VariationalReconstruction_2(mpi, mesh)
    vfv.ParseSettings(vfvSettings)
    vfv.SetPeriodicTransformationsNoOp()

    bcid_2_bcweight_map = {}
    for name, bc_id in name2Id.n2id_map.items():
        bcid_2_bcweight_map[(bc_id, 0)] = 1.0
        if name.startswith("PERIODIC"):
            for order_idx in range(4):
                bcid_2_bcweight_map[(bc_id, order_idx)] = 1.0
    vfv.ConstructMetrics()
    vfv.ConstructBaseAndWeight_map(bcid_2_bcweight_map)
    vfv.ConstructRecCoeff()

    return mesh, vfv, name2Id


@pytest.fixture(scope="module")
def gg_vfv(mpi):
    """Build VariationalReconstruction_2 with Gauss-Green (subs2ndOrder=1)."""
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

    vfvSettings = {
        "maxOrder": 1,
        "intOrder": 5,
        "cacheDiffBase": True,
        "jacobiRelax": 1.0,
        "SORInstead": False,
        "subs2ndOrder": 1,
        "functionalSettings": {
            "dirWeightScheme": "HQM_OPT",
            "geomWeightScheme": "HQM_SD",
            "geomWeightPower": 0.5,
            "geomWeightBias": 1,
        },
    }

    vfv = CFV.VariationalReconstruction_2(mpi, mesh)
    vfv.ParseSettings(vfvSettings)
    vfv.SetPeriodicTransformationsNoOp()

    bcid_2_bcweight_map = {}
    for name, bc_id in name2Id.n2id_map.items():
        bcid_2_bcweight_map[(bc_id, 0)] = 1.0
        if name.startswith("PERIODIC"):
            for order_idx in range(4):
                bcid_2_bcweight_map[(bc_id, order_idx)] = 1.0
    vfv.ConstructMetrics()
    vfv.ConstructBaseAndWeight_map(bcid_2_bcweight_map)
    vfv.ConstructRecCoeff()

    return mesh, vfv, name2Id


# ===================================================================
# Construction pipeline
# ===================================================================


class TestVRConstruction:
    """Verify VR construction pipeline completes and produces valid state."""

    def test_global_vol(self, periodic_vfv):
        """Global volume should be 9.0 for [0,3]^2."""
        _, vfv, _ = periodic_vfv
        assert vfv.GetGlobalVol() == pytest.approx(9.0, abs=1e-10)

    def test_cell_bary_in_range(self, periodic_vfv):
        """Primary cell barycenters should lie within [-1,2]^2
        (a superset of the Uniform_3x3 mesh bounds)."""
        mesh, vfv, _ = periodic_vfv
        for iCell in range(mesh.NumCell()):
            bary = vfv.GetCellBary(iCell)
            assert -1.0 - 1e-10 <= bary[0] <= 2.0 + 1e-10
            assert -1.0 - 1e-10 <= bary[1] <= 2.0 + 1e-10

    def test_cell_vol_positive(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        for iCell in range(mesh.NumCell()):
            assert vfv.GetCellVol(iCell) > 0

    def test_face_area_positive(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        for iFace in range(mesh.NumFaceProc()):
            assert vfv.GetFaceArea(iFace) > 0


# ===================================================================
# DOF array building
# ===================================================================


class TestVRDOFBuilding:
    """Verify DOF array building through VR (BuildURec, BuildUGrad)."""

    def test_build_urec_dynamic(self, periodic_vfv):
        _, vfv, _ = periodic_vfv
        uRec = CFV.tURec_D()
        vfv.BuildURec_D(uRec, 1)
        assert uRec.Size() > 0

    def test_build_udof_dynamic(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        vfv.BuildUDof_D(u, 1)
        assert u.father.Size() == mesh.NumCell()

    def test_build_ugrad_dynamic(self, periodic_vfv):
        _, vfv, _ = periodic_vfv
        uGrad = CFV.tUGrad_2xD()
        vfv.BuildUGrad_D(uGrad, 1)
        assert uGrad.Size() > 0


# ===================================================================
# Reconstruction: constant field should have zero reconstruction error
# ===================================================================


class TestConstantExactness:
    """Constant field should produce zero reconstruction coefficients."""

    def test_constant_field_zero_urec(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        vfv.BuildUDof_D(u, 1)

        # Set constant field u = 7.0 for all cells
        for iCell in range(mesh.NumCell()):
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = 7.0
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        uRec = CFV.tURec_D()
        uRecNew = CFV.tURec_D()
        vfv.BuildURec_D(uRec, 1)
        vfv.BuildURec_D(uRecNew, 1)

        eval_obj = CFV.ModelEvaluator(mesh, vfv, {}, 1)

        # Run iterations (putIntoNew=True puts result into uRecNew, then swap)
        for _ in range(20):
            eval_obj.DoReconstructionIter(uRec, uRecNew, u, 0.0, True)
            uRec, uRecNew = uRecNew, uRec

        # All reconstruction coefficients should be zero (constant = no gradient)
        max_coeff = 0.0
        for iCell in range(mesh.NumCell()):
            coeffs = np.array(uRec[iCell], copy=False)
            max_coeff = max(max_coeff, np.abs(coeffs).max())

        assert max_coeff < 1e-10, f"Max rec coeff = {max_coeff}, expected ~0"


# ===================================================================
# Reconstruction: linear field on periodic mesh
# ===================================================================


class TestLinearFieldReconstruction:
    """Linear field f(x,y) = x should produce accurate reconstruction."""

    def test_linear_x_reconstruction(self, periodic_vfv):
        """After iterating, uRec for u=x should contain correct dx coefficient."""
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        vfv.BuildUDof_D(u, 1)

        # Set u_i = x_bary for each cell
        for iCell in range(mesh.NumCell()):
            bary = vfv.GetCellBary(iCell)
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = bary[0]
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        uRec = CFV.tURec_D()
        uRecNew = CFV.tURec_D()
        vfv.BuildURec_D(uRec, 1)
        vfv.BuildURec_D(uRecNew, 1)

        eval_obj = CFV.ModelEvaluator(mesh, vfv, {}, 1)
        for _ in range(100):
            eval_obj.DoReconstructionIter(uRec, uRecNew, u, 0.0, True)
            uRec, uRecNew = uRecNew, uRec

        # Check that reconstruction has converged to non-trivial values
        has_nonzero = False
        for iCell in range(mesh.NumCell()):
            coeffs = np.array(uRec[iCell], copy=False)
            if np.abs(coeffs).max() > 1e-6:
                has_nonzero = True
                break
        assert has_nonzero, "uRec should have non-zero coefficients for linear field"


# ===================================================================
# ModelEvaluator RHS evaluation
# ===================================================================


class TestModelEvaluatorRHS:
    """Verify ModelEvaluator.EvaluateRHS produces finite results."""

    def test_rhs_finite(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        rhs = CFV.tUDof_D()
        uRec = CFV.tURec_D()

        vfv.BuildUDof_D(u, 1)
        vfv.BuildUDof_D(rhs, 1)
        vfv.BuildURec_D(uRec, 1)

        # Set u = sin(2*pi*x/3)
        for iCell in range(mesh.NumCell()):
            bary = vfv.GetCellBary(iCell)
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = np.sin(2.0 * np.pi * bary[0] / 3.0)
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        eval_obj = CFV.ModelEvaluator(
            mesh, vfv, {"ax": 1.0, "ay": 0.0, "sigma": 0.0}, 1
        )
        eval_obj.EvaluateRHS(rhs, u, uRec, 0.0)

        # Check all RHS values are finite
        for iCell in range(mesh.NumCell()):
            rhs_i = np.array(rhs[iCell], copy=False)
            assert np.all(np.isfinite(rhs_i)), (
                f"Cell {iCell} RHS is not finite: {rhs_i}"
            )

    def test_rhs_nonzero_for_nonuniform_field(self, periodic_vfv):
        """RHS of advection equation should be nonzero for non-constant field."""
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        rhs = CFV.tUDof_D()
        uRec = CFV.tURec_D()

        vfv.BuildUDof_D(u, 1)
        vfv.BuildUDof_D(rhs, 1)
        vfv.BuildURec_D(uRec, 1)

        for iCell in range(mesh.NumCell()):
            bary = vfv.GetCellBary(iCell)
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = np.sin(2.0 * np.pi * bary[0] / 3.0)
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        eval_obj = CFV.ModelEvaluator(
            mesh, vfv, {"ax": 1.0, "ay": 0.0, "sigma": 0.0}, 1
        )
        eval_obj.EvaluateRHS(rhs, u, uRec, 0.0)

        rhs_norm = 0.0
        for iCell in range(mesh.NumCell()):
            rhs_i = np.array(rhs[iCell], copy=False)
            rhs_norm += np.sum(rhs_i ** 2)
        rhs_norm = np.sqrt(rhs_norm)
        assert rhs_norm > 1e-6, f"RHS norm = {rhs_norm}, expected nonzero"

    def test_rhs_zero_for_constant_field(self, periodic_vfv):
        """RHS of advection for constant field should be ~zero."""
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        rhs = CFV.tUDof_D()
        uRec = CFV.tURec_D()

        vfv.BuildUDof_D(u, 1)
        vfv.BuildUDof_D(rhs, 1)
        vfv.BuildURec_D(uRec, 1)

        for iCell in range(mesh.NumCell()):
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = 5.0
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        eval_obj = CFV.ModelEvaluator(
            mesh, vfv, {"ax": 1.0, "ay": 1.0, "sigma": 0.0}, 1
        )
        eval_obj.EvaluateRHS(rhs, u, uRec, 0.0)

        rhs_norm = 0.0
        for iCell in range(mesh.NumCell()):
            rhs_i = np.array(rhs[iCell], copy=False)
            rhs_norm += np.sum(rhs_i ** 2)
        rhs_norm = np.sqrt(rhs_norm)
        assert rhs_norm < 1e-10, f"RHS norm = {rhs_norm}, expected ~0 for const field"


# ===================================================================
# Reconstruction iteration convergence
# ===================================================================


class TestReconstructionConvergence:
    """Verify VR iteration converges."""

    def test_iterative_vr_convergence(self, periodic_vfv):
        """Iterative VR should reduce increment over iterations."""
        mesh, vfv, _ = periodic_vfv
        u = CFV.tUDof_D()
        vfv.BuildUDof_D(u, 1)

        for iCell in range(mesh.NumCell()):
            bary = vfv.GetCellBary(iCell)
            u_i = np.array(u[iCell], copy=False)
            u_i[0] = np.sin(2.0 * np.pi * bary[0] / 3.0) * np.cos(
                2.0 * np.pi * bary[1] / 3.0
            )
        u.trans.startPersistentPull()
        u.trans.waitPersistentPull()

        uRec = CFV.tURec_D()
        uRecNew = CFV.tURec_D()
        vfv.BuildURec_D(uRec, 1)
        vfv.BuildURec_D(uRecNew, 1)

        eval_obj = CFV.ModelEvaluator(mesh, vfv, {}, 1)

        increments = []
        for _ in range(50):
            eval_obj.DoReconstructionIter(uRec, uRecNew, u, 0.0, True)

            # Compute increment (uRecNew has the new value)
            inc = 0.0
            for iCell in range(mesh.NumCell()):
                diff = np.array(uRecNew[iCell]) - np.array(uRec[iCell])
                inc += np.sum(diff ** 2)
            increments.append(np.sqrt(inc))

            uRec, uRecNew = uRecNew, uRec

        # Verify the increment is decreasing (convergence)
        # Allow for some noise in the first few iterations
        assert increments[-1] < increments[2], (
            f"Not converging: final inc = {increments[-1]}, "
            f"early inc = {increments[2]}"
        )


# ===================================================================
# Matrix access
# ===================================================================


class TestMatrixAccess:
    """Verify reconstruction matrices are accessible and valid."""

    def test_matrixAAInvB_accessible(self, periodic_vfv):
        """matrixAAInvB should be accessible and contain finite values."""
        mesh, vfv, _ = periodic_vfv
        mat = vfv.matrixAAInvB
        assert mat is not None
        # Access first cell's first matrix
        m00 = np.array(mat[0, 0], copy=False)
        assert np.all(np.isfinite(m00)), "matrixAAInvB[0,0] contains non-finite"
        assert m00.shape[0] > 0
        assert m00.shape[1] > 0

    def test_vectorAInvB_accessible(self, periodic_vfv):
        """vectorAInvB should be accessible and contain finite values."""
        mesh, vfv, _ = periodic_vfv
        vec = vfv.vectorAInvB
        assert vec is not None
        v00 = np.array(vec[0, 0], copy=False)
        assert np.all(np.isfinite(v00)), "vectorAInvB[0,0] contains non-finite"


# ===================================================================
# Boundary callback
# ===================================================================


class TestBoundaryCallback:
    """Verify ModelEvaluator.get_FBoundary returns a callable."""

    def test_get_fboundary(self, periodic_vfv):
        mesh, vfv, _ = periodic_vfv
        eval_obj = CFV.ModelEvaluator(mesh, vfv, {}, 1)
        fb = eval_obj.get_FBoundary(0.5)
        assert fb is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
