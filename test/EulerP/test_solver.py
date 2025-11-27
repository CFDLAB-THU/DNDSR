from DNDSR import DNDS, Geom, CFV, EulerP
import json
from DNDSR.Geom.utils import *
import numpy as np

from DNDSR.EulerP.EulerP_Solver import Solver
import cProfile, time


def test_solver(
    mpi: DNDS.MPIInfo,
    ifCuda=False,
):
    if ifCuda:
        import cupy as cp

        # cp.cuda.Device(1).use()

    m_name, dim = "Uniform128_Periodic.cgns", 2
    # m_name, dim = "UP3D_128.cgns", 3
    meshFile = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "mesh",
        m_name,
    )
    outDir = "../data/out/test3"

    solver = Solver(mpi)
    solver.ReadMesh(
        meshFile,
        dim=dim,
        other_options={
            "meshDirectBisect": 3,
            "second_level_parts": int(128),
        },
    )

    solver.BuildFV(
        fv_settings={"intOrder": 1, "maxOrder": 1},
    )
    solver.BuildEval()
    solver.WrapEval()
    solver.BuildDataArray()
    solver.eval.setConfig(
        {
            "threadsPerBlock": 32,
        }
    )
    data = solver.data
    if ifCuda:
        solver.to_device("CUDA")

    u = solver.data["u"]
    mesh = solver.mesh
    numCellGlobal = mesh.NumCellGlobal()
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

    device_run = DNDS.DeviceBackend.Unknown
    if ifCuda:
        solver.data_to_device("CUDA")
        device_run = DNDS.DeviceBackend.CUDA
        for n, a in data.items():
            a.trans.initPersistentPull(device_run)

    solver.runningDevice = device_run
    u.trans.startPersistentPull(device_run)
    u.trans.waitPersistentPull(device_run)

    # print(f"son size {u.son.Size()}")
    # for i in range(100000):
    #     tW0 = time.perf_counter()
    #     u.trans.startPersistentPull(device_run)
    #     u.trans.waitPersistentPull(device_run)
    #     tW1 = time.perf_counter()
    #     if mpi.rank == 0:
    #         print(f"time: {tW1 - tW0:.3e}")

    # 0.5, 1, 0, 0, 4

    tInt = 0.01
    nInt = 1
    t = 0
    pr = cProfile.Profile()
    pr.enable()

    iStep = 0

    for II in range(1, nInt + 1):
        tW0 = time.perf_counter()
        iStepNew, tNew = solver.IntegrateDt_ExplicitInterval(
            t, t + tInt, CFL=0.5, step0=iStep, max_step=iStep + 100000
        )
        tW1 = time.perf_counter()
        if mpi.rank == 0:
            print("=" * 32)
            print()
            tStep = (tW1 - tW0) / (iStepNew - iStep)
            print(f"Average time per step: [{tStep:.4e}]")
            print(f"cell step / second: [{numCellGlobal / tStep:.4e}]")
            print()
            print("=" * 32)
        t = tNew
        iStep = iStepNew

        data["uPrim"].to_host()
        data["p"].to_host()
        solver.eval.PrintDataVTKHDF(
            os.path.join(outDir, f"testC_{II}"),
            os.path.join(outDir, "testC"),
            arrCellCentScalar=[data["p"]],
            arrCellCentScalar_names=["p"],
            uPrimCell=data["uPrim"],
            t=t,
        )
        if ifCuda:
            data["uPrim"].to_device("CUDA")
            data["p"].to_device("CUDA")
    pr.disable()
    pr.dump_stats(f"profile_rank_{mpi.rank}.prof")


if __name__ == "__main__":
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    ifCuda = True
    if ifCuda:
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"{mpi.rank}, nGPU={device_count}")
        print(cp.cuda.Device(0))
        if mpi.size > device_count:
            raise RuntimeError("too many ranks on this machine")
        cp.cuda.Device(mpi.rank).use()

    test_solver(mpi, ifCuda=ifCuda)
