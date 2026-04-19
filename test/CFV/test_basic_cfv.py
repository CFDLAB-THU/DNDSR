from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import *
import numpy as np
import json


def test_basic():
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()

    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    )

    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "Uniform_3x3.cgns"
    )

    mesh, reader, name2Id = create_mesh_from_CGNS(
        meshFile,
        mpi,
        2,
        periodic_geometry={
            "translation1": [3, 0, 0],
            "translation2": [0, 3, 0],
        },
    )

    meshBnd, readerBnd = create_bnd_mesh(mesh)

    vfvSettings = json.loads(
        "".join(
            [
                line if not line.strip().startswith("//") else ""
                for line in """{
        "maxOrder": 3,
        "intOrder": 5,
        "intOrderVR": 5,
        "cacheDiffBase": true,
        "jacobiRelax": 1.0,
        "SORInstead": false,
        "smoothThreshold": 1e-3,
        "WBAP_nStd": 10.0,
        "normWBAP": false,
        "subs2ndOrder": 1,
        "subs2ndOrderGGScheme": 0,
        "baseSettings": {
            "localOrientation": false,
            "anisotropicLengths": false
        },
        "functionalSettings": {
            // "scaleType": "MeanAACBB",
            "dirWeightScheme": "HQM_OPT",
            // "dirWeightScheme": "ManualDirWeight",
            // "manualDirWeights": [
            //     1.0,
            //     1,
            //     0,
            //     0
            // ],
            "geomWeightScheme": "HQM_SD",
            "geomWeightPower": 0.5,
            "geomWeightBias": 1,
            // "geomWeightScheme": "SD_Power",
            // "geomWeightPower1": -0.5,
            // "geomWeightPower2": 0.5,
            // "useAnisotropicFunctional": true,
            // // "anisotropicType": "InertiaCoordBB",
            // "inertiaWeightPower": 0,
            // "scaleMultiplier": 1,
            "_tail": 0
        }
    }
    """.splitlines()
            ]
        )
    )
    print(vfvSettings)

    vfv = CFV.VariationalReconstruction_2(mpi, mesh)
    vfv.ParseSettings(vfvSettings)
    vfv.SetPeriodicTransformationsNoOp()

    bcid_2_bcweight_map = {}
    for name, id in name2Id.n2id_map.items():
        # if name == "WALL":
        bcid_2_bcweight_map[(id, 0)] = 1.0
        if name.startswith("PERIODIC"):
            bcid_2_bcweight_map[(id, 0)] = 1.0
            bcid_2_bcweight_map[(id, 1)] = 1.0
            bcid_2_bcweight_map[(id, 2)] = 1.0
            bcid_2_bcweight_map[(id, 3)] = 1.0
    vfv.ConstructMetrics()
    vfv.ConstructBaseAndWeight_map(bcid_2_bcweight_map)
    vfv.ConstructRecCoeff()

    u = CFV.tUDof_D()
    rhs = CFV.tUDof_D()
    uRec, uRecNew = [CFV.tURec_D() for _ in range(2)]
    for u_ in [u, rhs]:
        vfv.BuildUDof_D(u_, 1)
    for uRec_ in [uRec, uRecNew]:
        vfv.BuildURec_D(uRec_, 1)
    for i in range(mesh.NumCell()):
        u_i = np.array(u[i], copy=False)
        u_i[0] = vfv.GetCellBary(i)[0]
        print(vfv.GetCellBary(i))
        print(np.array(u[i], copy=False))

    print(
        f"rank [{mpi.rank}], u.Size() = {u.Size()}, u.father.Size() = {u.father.Size()}"
        + "\n"
        + f"{u.trans.mpi}"
    )
    eval = CFV.ModelEvaluator(mesh, vfv, {}, 1)
    eval.EvaluateRHS(rhs, u, uRec, 0.0)
    for i in range(mesh.NumCell()):
        print(np.array(rhs[i], copy=False))
    FBoundary = eval.get_FBoundary(0.5)
    eval.DoReconstructionIter(uRec, uRecNew, u, 0.0, False)
    for i in range(mesh.NumCell()):
        print(np.array(uRec[i], copy=False))


if __name__ == "__main__":
    test_basic()
