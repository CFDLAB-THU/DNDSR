from DNDSR import DNDS, Geom, CFV, EulerP
import json
from DNDSR.Geom.utils import *
import numpy as np

from DNDSR.EulerP.EulerP_Solver import Solver


def test_solver(
    mpi: DNDS.MPIInfo,
):
    meshFile = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "mesh",
        "Uniform128_Periodic.cgns",
    )

    solver = Solver(mpi)
    solver.ReadMesh(
        meshFile,
        dim=2,
        other_options={
            "meshDirectBisect": 2,
        },
    )
    solver.BuildFV(
        fv_settings={"intOrder": 1, "maxOrder": 1},
    )
    solver.BuildEval()
    solver.BuildDataArray()

    u = solver.data["u"]
    mesh = solver.mesh
    fv = solver.fv
    u.setConstant(np.array([1, 0, 0, 0, 2.5]).reshape(-1, 1))
    for iCell in range(mesh.NumCell()):
        x = fv.GetCellBary(iCell)
        if np.linalg.norm(np.array([0.5, 0.5, 0]) - x, np.inf) < 0.25:
            # if abs(x[0] - 0.5) < 0.25:
            v = 0.1
            uBox = [0.5, 1, 0, 0, 4]
            # uBox = [1, 1 * v, 0, 0, 2.5 + 0.5 * 1 * v**2]
            u[iCell] = np.array(uBox, dtype=np.float64).reshape(-1, 1)
        # uu = 1
        # vv = 1
        # r = 1 + 0.1 * np.cos(x[0] * 2 * np.pi) * np.cos(x[1] * 2 * np.pi)
        # u[iCell] = np.array(
        #     [r, r * uu, r * vv, 0, 2.5 + 0.5 * r * (uu**2 + vv**2)], dtype=np.float64
        # ).reshape(-1, 1)
    u.trans.startPersistentPull()
    u.trans.waitPersistentPull()

    # 0.5, 1, 0, 0, 4

    data = solver.data

    tInt = 0.01
    nInt = 40
    t = 0
    for II in range(1, nInt + 1):
        solver.IntegrateDt_ExplicitInterval(t, t + tInt, CFL=0.01)
        t += tInt
        solver.eval.PrintDataVTKHDF(
            f"testC_{II}",
            "testC",
            arrCellCentScalar=[data["p"]],
            arrCellCentScalar_names=["p"],
            uPrimCell=data["uPrim"],
            t=t,
        )


if __name__ == "__main__":
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    test_solver(mpi)
