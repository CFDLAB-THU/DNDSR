from DNDSR import DNDS, Geom, CFV, EulerP
import json
from DNDSR.Geom.utils import *
import numpy as np
import debugpy

from DNDSR.DNDS.Debug_Py import MPIDebugHold


def get_fv(mpi):
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()
    # meshFile = os.path.join(
    #     os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    # )
    # meshFile = os.path.join(
    #     os.path.dirname(__file__), "..", "..", "data", "mesh", "Uniform_3x3.cgns"
    # )
    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "Uniform32_Periodic.cgns"
    )
    mesh, reader, name2Id = create_mesh_from_CGNS(
        meshFile,
        mpi,
        2,
        periodic_geometry={
            # "translation1": [3, 0, 0],
            # "translation2": [0, 3, 0],
            "translation1": [1, 0, 0],
            "translation2": [0, 1, 0],
        },
        meshDirectBisect=4,
        second_level_parts=16,
    )
    meshBnd, readerBnd = create_bnd_mesh(mesh)

    fv = CFV.FiniteVolume(mpi, mesh)
    settings = fv.GetSettings()
    settings["intOrder"] = 3
    settings["maxOrder"] = 3
    fv.ParseSettings(settings)
    if mpi.rank == 0:
        print(fv.GetSettings())

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

    return mesh, reader, name2Id, meshBnd, readerBnd, fv


def test_basic_eulerP(mpi: DNDS.MPIInfo, isCuda=False):

    mesh, reader, name2Id, meshBnd, readerBnd, fv = get_fv(mpi)
    print(name2Id.n2id_map)
    print(name2Id.n2id_map is name2Id.n2id_map)
    phys_dict = EulerP.Physics().to_dict()
    phys_params = phys_dict["params"]
    phys_params["TRef"] = 273.15
    phys_params["cp"] = 1.0
    phys_params["gamma"] = 1.4
    phys_params["mu0"] = 1e-10
    phys_dict["reference_values"] = [1.0, 1.0, 1.0, 1.0, 1.0]

    phys = EulerP.Physics.from_dict(phys_dict)
    print("Physics:")
    print(phys.to_dict())

    # bcInputs = [EulerP.BCInput()]
    # bcInputs[0].name = "wall"
    # bcInputs[0].type = EulerP.BCType.Wall
    # bcInputs[0].value = [1]

    bcInputs = []

    bcHandler = EulerP.BCHandler(bcInputs, name2Id)
    print(bcHandler.id2bc(1).type)

    eval = EulerP.Evaluator(fv, bcHandler, phys)

    u = CFV.tUDof_5()
    fv.BuildUDof_5(u, 5)
    gU = CFV.tUGrad_3x5()
    fv.BuildUGrad_3x5(gU, 5)
    s = CFV.tUDof_1()
    fv.BuildUDof_1(s, 1)
    gS = CFV.tUGrad_3x1()
    fv.BuildUGrad_3x1(gS, 1)

    uf = CFV.tUDof_5()
    fv.BuildUDof_5(uf, 5, varloc=Geom.MeshLoc.Face)
    gUf = CFV.tUGrad_3x5()
    fv.BuildUGrad_3x5(gUf, 5, varloc=Geom.MeshLoc.Face)
    sf = CFV.tUDof_1()
    fv.BuildUDof_1(sf, 1, varloc=Geom.MeshLoc.Face)
    gSf = CFV.tUGrad_3x1()
    fv.BuildUGrad_3x1(gSf, 1, varloc=Geom.MeshLoc.Face)
    if isCuda:
        mesh.to_device("CUDA")
        fv.to_device("CUDA")
        eval.to_device("CUDA")
        eval_device = eval.device()
        print(eval_device)

        u.to_device("CUDA")
        gU.to_device("CUDA")
        s.to_device("CUDA")
        gS.to_device("CUDA")

        uf.to_device("CUDA")
        gUf.to_device("CUDA")
        sf.to_device("CUDA")
        gSf.to_device("CUDA")

    data = {}
    data["u"] = u
    data["uGrad"] = gU

    u.setConstant(np.array([1, 0, 0, 0, 2.5]).reshape(-1, 1))
    for iCell in range(mesh.NumCell()):
        x = fv.GetCellBary(iCell)
        if np.linalg.norm(np.array([0.5, 0.5, 0]) - x, 2) < 0.25:
            u[iCell] = np.array([1, 4, 0, 0, 10.5], dtype=np.float64).reshape(-1, 1)

        uu = 1
        vv = 1
        r = 1 + 0.1 * np.cos(x[0] * 2 * np.pi) * np.cos(x[1] * 2 * np.pi)
        u[iCell] = np.array(
            [r, r * uu, r * vv, 0, 2.5 + 0.5 * r * (uu**2 + vv**2)], dtype=np.float64
        ).reshape(-1, 1)
    if isCuda:
        u.to_device("CUDA")
    print(u.father)

    data["uGrad"].setConstant(100.0)
    print(data["uGrad"].norm2())
    RecGradient_Arg = eval.RecGradient_Arg()
    RecGradient_Arg.u = data["u"]
    RecGradient_Arg.uGrad = data["uGrad"]
    RecGradient_Arg.uScalar = []
    RecGradient_Arg.uScalarGrad = []
    eval.RecGradient(RecGradient_Arg)

    data["p"] = s.clone()
    data["T"] = s.clone()
    data["a"] = s.clone()
    data["mu"] = s.clone()
    data["gamma"] = s.clone()

    data["uPrim"] = u.clone()
    data["uGradPrim"] = gU.clone()

    Cons2PrimMu_Arg = eval.Cons2PrimMu_Arg()
    Cons2PrimMu_Arg.p = data["p"]
    Cons2PrimMu_Arg.T = data["T"]
    Cons2PrimMu_Arg.a = data["a"]
    Cons2PrimMu_Arg.mu = data["mu"]
    Cons2PrimMu_Arg.gamma = data["gamma"]

    Cons2PrimMu_Arg.u = data["u"]
    Cons2PrimMu_Arg.uGrad = data["uGrad"]
    Cons2PrimMu_Arg.uPrim = data["uPrim"]
    Cons2PrimMu_Arg.uGradPrim = data["uGradPrim"]

    eval.Cons2PrimMu(Cons2PrimMu_Arg)

    data["deltaLamCell"] = s.clone()
    data["dt"] = s.clone()
    data["uFL"] = uf
    data["uGradFF"] = gUf
    data["deltaLamFace"] = sf.clone()
    data["faceLamEst"] = gSf.clone()
    data["faceLamVisEst"] = sf.clone()

    EstEigenDt_Arg = eval.EstEigenDt_Arg()
    EstEigenDt_Arg.u = data["u"]
    EstEigenDt_Arg.muCell = data["mu"]
    EstEigenDt_Arg.aCell = data["a"]
    EstEigenDt_Arg.deltaLamCell = data["deltaLamCell"]
    EstEigenDt_Arg.deltaLamFace = data["deltaLamFace"]
    EstEigenDt_Arg.faceLamEst = data["faceLamEst"]
    EstEigenDt_Arg.faceLamVisEst = data["faceLamVisEst"]
    EstEigenDt_Arg.dt = data["dt"]
    eval.EstEigenDt(EstEigenDt_Arg)

    data["uGradFF"] = gUf.clone()
    data["uFL"] = uf.clone()
    data["uFR"] = uf.clone()

    RecFace2nd_Arg = eval.RecFace2nd_Arg()
    RecFace2nd_Arg.u = data["u"]
    RecFace2nd_Arg.uGrad = data["uGrad"]
    RecFace2nd_Arg.uFL = data["uFL"]
    RecFace2nd_Arg.uFR = data["uFR"]
    RecFace2nd_Arg.uGradFF = data["uGradFF"]
    # data["uGrad"].setConstant(0.0)
    eval.RecFace2nd(RecFace2nd_Arg)

    data["uPrimFL"] = uf.clone()
    data["pFL"] = sf.clone()
    data["TFL"] = sf.clone()
    data["aFL"] = sf.clone()
    data["gammaFL"] = sf.clone()
    data["uPrimFR"] = uf.clone()
    data["pFR"] = sf.clone()
    data["TFR"] = sf.clone()
    data["aFR"] = sf.clone()
    data["gammaFR"] = sf.clone()

    Cons2Prim_Arg_FL = eval.Cons2Prim_Arg()
    Cons2Prim_Arg_FL.u = data["uFL"]
    Cons2Prim_Arg_FL.uPrim = data["uPrimFL"]
    Cons2Prim_Arg_FL.p = data["pFL"]
    Cons2Prim_Arg_FL.T = data["TFL"]
    Cons2Prim_Arg_FL.a = data["aFL"]
    Cons2Prim_Arg_FL.gamma = data["gammaFL"]

    Cons2Prim_Arg_FR = eval.Cons2Prim_Arg()
    Cons2Prim_Arg_FR.u = data["uFR"]
    Cons2Prim_Arg_FR.uPrim = data["uPrimFR"]
    Cons2Prim_Arg_FR.p = data["pFR"]
    Cons2Prim_Arg_FR.T = data["TFR"]
    Cons2Prim_Arg_FR.a = data["aFR"]
    Cons2Prim_Arg_FR.gamma = data["gammaFR"]
    eval.Cons2Prim(Cons2Prim_Arg_FL)
    eval.Cons2Prim(Cons2Prim_Arg_FR)

    pFMin = min(data["pFR"].min(), data["pFL"].min())
    print(f"pFMin, {pFMin}")

    data["fluxFF"] = uf.clone()
    data["rhs"] = u.clone()

    Flux2nd_Arg = eval.Flux2nd_Arg()
    Flux2nd_Arg.u = data["u"]
    Flux2nd_Arg.uGrad = data["uGrad"]
    Flux2nd_Arg.uPrim = data["uPrim"]
    Flux2nd_Arg.uGradPrim = data["uGradPrim"]

    Flux2nd_Arg.p = data["p"]
    Flux2nd_Arg.T = data["T"]
    Flux2nd_Arg.a = data["a"]
    Flux2nd_Arg.mu = data["mu"]
    Flux2nd_Arg.gamma = data["gamma"]

    Flux2nd_Arg.uFL = data["uFL"]
    Flux2nd_Arg.uFR = data["uFR"]
    Flux2nd_Arg.pFL = data["pFL"]
    Flux2nd_Arg.pFR = data["pFR"]

    Flux2nd_Arg.uGradFF = data["uGradFF"]
    Flux2nd_Arg.deltaLamFaceFF = data["deltaLamFace"]

    Flux2nd_Arg.fluxFF = data["fluxFF"]
    Flux2nd_Arg.rhs = data["rhs"]

    data["rhs"].setConstant(0.0)
    eval.Flux2nd(Flux2nd_Arg)

    check_norm = data["p"].norm2()
    print(f"p norm, {check_norm}")
    check_norm = data["p"].min()
    print(f"p min, {check_norm}")

    data["p"].to_host()
    check_norm = data["p"].min()
    print(f"p min, {check_norm}")

    for n, a in data.items():
        a.to_host()
    cell_out = s.clone()
    cell_out1 = s.clone()
    cell_out2 = s.clone()

    for iCell in range(mesh.NumCell()):
        np.array(cell_out[iCell], copy=False)[:] = np.array(
            data["rhs"][iCell], copy=False
        )[0]
        np.array(cell_out1[iCell], copy=False)[:] = np.array(
            data["u"][iCell], copy=False
        )[1]
        np.array(cell_out1[iCell], copy=False)[:] = iCell
        np.array(cell_out2[iCell], copy=False)[:] = np.array(
            data["uGrad"][iCell], copy=False
        )[0, 1]

    eval.PrintDataVTKHDF(
        "test_0",
        "test",
        [cell_out, cell_out1, cell_out2, data["p"]],
        ["cell_out", "cell_out1", "cell_out2", "p"],
    )

    # for iCell in range(mesh.NumCell()):
    #     print(iCell)
    #     print(f"{np.asarray(data['u'][iCell]).T}")
    #     for iFace in mesh.cell2face[iCell]:
    #         print(f"{iFace} -- {np.asarray(data['uFL'][iFace]).T}")


if __name__ == "__main__":
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    ## debug py

    # MPIDebugHold()

    # DNDS.Debug.MPIDebugHold(mpi)
    test_basic_eulerP(mpi, isCuda=False)
