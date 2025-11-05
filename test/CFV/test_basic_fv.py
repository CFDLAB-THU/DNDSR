from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import *
import numpy as np
import json
import time


import time


def time_function_until_limit(
    func, time_limit=1.0, max_executions=None, *args, **kwargs
):
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_executions is not None and max_executions <= 0:
        raise ValueError("max_executions must be positive if provided")

    executions = 0
    start_time = time.perf_counter()

    while True:
        # Check if we've hit the max executions
        if max_executions is not None and executions >= max_executions:
            break

        # Execute the function
        func(*args, **kwargs)
        executions += 1

        # Check elapsed time
        elapsed = time.perf_counter() - start_time
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
        meshDirectBisect=1,
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

    mesh.to_device("CUDA")
    fv.to_device("CUDA")

    u = CFV.tUDof_D()
    grad_u = CFV.tUGrad_3xD()

    nvars = 5

    fv.BuildUDof_D(u, nvars)
    fv.BuildUGrad_3xD(grad_u, nvars)

    for iCell in range(mesh.NumCell()):
        x = fv.GetCellBary(iCell)
        ui = np.array(u[iCell], copy=False)
        ui[:] = x[0] + np.sin(x[1] * np.pi)
    u.trans.startPersistentPull()
    u.trans.waitPersistentPull()

    u.to_device("CUDA")
    grad_u.to_device("CUDA")

    def test_CUDA():
        CFV.finiteVolumeCellOpTest_main_CUDA(fv, u, grad_u)

    def test_Host():
        CFV.finiteVolumeCellOpTest_main_Host(fv, u, grad_u)

    executions, total_time, avg_time = time_function_until_limit(test_Host, 1.0, 100000)
    print("--- HOST ---")
    print(f"[{executions}] times, avg [{avg_time:.4g}] s")
    avg_time_host = avg_time
    executions, total_time, avg_time = time_function_until_limit(test_CUDA, 1.0, 100000)
    print("--- CUDA ---")
    print(f"[{executions}] times, avg [{avg_time:.4g}] s")
    print(f" -- acc [{avg_time_host / avg_time:.4g}]")


if __name__ == "__main__":
    test_basic()
