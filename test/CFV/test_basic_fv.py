from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import *
import numpy as np
import json
import time
import pprint

import time


def time_function_until_limit(
    func,
    time_limit=1.0,
    max_executions=None,
    iter_pack=1,
    report=None,
    *args,
    **kwargs,
):
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_executions is not None and max_executions <= 0:
        raise ValueError("max_executions must be positive if provided")

    executions = 0
    start_time = time.perf_counter()

    reportTime = 1

    while True:
        # Check if we've hit the max executions
        if max_executions is not None and executions >= max_executions:
            break

        # Execute the function
        for _ in range(iter_pack):
            func(*args, **kwargs)
        executions += iter_pack

        # Check elapsed time
        elapsed = time.perf_counter() - start_time
        if elapsed >= reportTime:
            reportTime += 1
            if report is not None:
                report(elapsed, executions)
        if elapsed >= time_limit:
            break

    total_time = time.perf_counter() - start_time
    avg_time = total_time / executions if executions > 0 else 0.0

    return (executions, total_time, avg_time)


def test_basic():
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()

    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    )

    # meshFile = os.path.join(
    #     os.path.dirname(__file__), "..", "..", "data", "mesh", "Uniform_3x3.cgns"
    # )

    mesh, reader, name2Id = create_mesh_from_CGNS(
        meshFile,
        mpi,
        2,
        periodic_geometry={
            "translation1": [3, 0, 0],
            "translation2": [0, 3, 0],
        },
        meshDirectBisect=2,
    )

    meshBnd, readerBnd = create_bnd_mesh(mesh)

    fv = CFV.FiniteVolume(mpi, mesh)
    settings = fv.GetSettings()
    settings["intOrder"] = 5
    settings["maxOrder"] = 3
    fv.ParseSettings(settings)
    if mpi.rank == 0:
        print(fv.GetSettings())

    bcid_2_bcweight_map = {}
    for name, id in name2Id.n2id_map.items():
        # if name == "WALL":
        bcid_2_bcweight_map[(id, 0)] = 1.0
        if name.startswith("PERIODIC"):
            bcid_2_bcweight_map[(id, 0)] = 1.0
            bcid_2_bcweight_map[(id, 1)] = 1.0
            bcid_2_bcweight_map[(id, 2)] = 1.0
            bcid_2_bcweight_map[(id, 3)] = 1.0

    # construction: volume
    fv.SetCellAtrBasic()
    fv.ConstructCellVolume()
    fv.ConstructCellBary()
    fv.ConstructCellCent()
    fv.ConstructCellIntJacobiDet()
    fv.ConstructCellIntPPhysics()
    fv.ConstructCellAlignedHBox()
    fv.ConstructCellMajorHBoxCoordInertia()
    # construction: face
    fv.SetFaceAtrBasic()
    fv.ConstructFaceCent()
    fv.ConstructFaceArea()
    fv.ConstructFaceIntJacobiDet()
    fv.ConstructFaceIntPPhysics()
    fv.ConstructFaceUnitNorm()
    fv.ConstructFaceMeanNorm()

    fv.ConstructCellSmoothScale()

    nB = fv.getArrayBytes() / 1024**2
    nBMesh = mesh.getArrayBytes() / 1024**2
    if mpi.rank == 0:
        print(f"Bytes     : {nB:.4g} MB")
        print(f"Bytes mesh: {nBMesh:.4g} MB")

    u = CFV.tUDof_D()
    grad_u_arrs = [CFV.tUGrad_3xD() for _ in range(2)];
    grad_u, grad_u1 = grad_u_arrs

    nvars = 5
    test_time = 4.0
    max_iter = 1000 * 1000 * 1000

    fv.BuildUDof_D(u, nvars)
    for arr in grad_u_arrs:
        fv.BuildUGrad_3xD(arr, nvars)
    u.setConstant(1.23)
    for iCell in range(mesh.NumCell()):
        x = fv.GetCellBary(iCell)
        ui = np.array(u[iCell], copy=False)
        ui[:] = x[0] + np.sin(x[1] * np.pi)
    u.trans.startPersistentPull()
    u.trans.waitPersistentPull()

    def test_CUDA():
        CFV.finiteVolumeCellOpTest_main_CUDA(
            fv,
            u,
            grad_u,
            {
                "threadsPerBlock": 256,
                "method": "pervar",
            },
        )

    def test_Host():
        CFV.finiteVolumeCellOpTest_main_Host(fv, u, grad_u)

    print("AAA0")
    grad_u.setConstant(0)
    print("AAA1")
    grad_u_norm2 = grad_u.norm2()
    if mpi.rank == 0:
        print(f"norm: {grad_u_norm2}")
    executions, total_time, avg_time = time_function_until_limit(
        test_Host,
        test_time,
        max_iter,
        iter_pack=100,
        report=lambda t, n: print(f" Host iter [{n:8}] time [{t:10.4e}]"),
    )
    grad_u_norm2 = grad_u.norm2()
    grad_u_cnorm1 = grad_u.componentWiseNorm1()
    if mpi.rank == 0:
        print("--- HOST ---")
        print(f"[{executions}] times, avg [{avg_time:8.04e}] s")
        print(f"norm: {grad_u.norm2()}")
    avg_time_host = avg_time

    grad_u1.assign_value(grad_u)
    grad_u.setConstant(0)

    mesh.to_device("CUDA")
    fv.to_device("CUDA")
    u.to_device("CUDA")
    grad_u.to_device("CUDA")
    grad_u1.to_device("CUDA")

    executions, total_time, avg_time = time_function_until_limit(
        test_CUDA,
        test_time,
        max_iter,
        iter_pack=100,
        report=lambda t, n: print(f" CUDA iter [{n:8}] time [{t:8.04e}]"),
    )
    # grad_u.to_host()
    grad_u_norm = grad_u.norm2()
    grad_u_cnorm1 = grad_u.componentWiseNorm1()
    grad_u1 *= np.ones((3, nvars))
    grad_u_err_norm = grad_u.norm2(grad_u1)
    grad_u_err_cnorm1 = grad_u.componentWiseNorm1(grad_u1)
    if mpi.rank == 0:
        print("--- CUDA ---")
        print(f"[{executions}] times, avg [{avg_time:.4g}] s")
        print(f"norm: {grad_u_norm}, diff_norm: {grad_u_err_norm:.4e}")
        pprint.pprint(grad_u_err_cnorm1.tolist())
        print(f" -- acc [{avg_time_host / avg_time:.4g}]")


if __name__ == "__main__":
    test_basic()
