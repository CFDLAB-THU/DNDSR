from DNDSR import DNDS, Geom, CFV, EulerP
import json
from DNDSR.Geom.utils import *


def get_fv(mpi):
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()
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
        meshDirectBisect=0,
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


def test_basic_eulerP(mpi: DNDS.MPIInfo):
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

    cell_out = CFV.tUDof_1()
    fv.BuildUDof_1(cell_out, 1, True, False)
    cell_out.setConstant(1.2)

    eval.PrintDataVTKHDF(
        "test_0", "test", [cell_out], ["cell_out"], [], [], [], [], [], [], 0.0
    )


if __name__ == "__main__":
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    test_basic_eulerP(mpi)
