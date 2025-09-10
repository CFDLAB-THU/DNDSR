import DNDSR
# print(DNDSR.__dict__)

from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import *
import numpy as np
import json



class WaveTester:
    def __init__(self, mpi):
        self.mpi = mpi

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
            "jacobiRelax": 1,
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

        self.mesh = mesh
        self.vfv = vfv
        self.eval = CFV.ModelEvaluator(mesh, vfv, {})

        u_real, rhs_real = [CFV.tUDof_1() for _ in range(2)]
        uRec_real, uRecNew_real = [CFV.tURec_1() for _ in range(2)]
        u_imag, rhs_imag = [CFV.tUDof_1() for _ in range(2)]
        uRec_imag, uRecNew_imag = [CFV.tURec_1() for _ in range(2)]
        self.u_list = [u_real, rhs_real, u_imag, rhs_imag]
        self.uRec_list = [uRec_real, uRecNew_real, uRec_imag, uRecNew_imag]

        for u_ in self.u_list:
            vfv.BuildUDof_1(u_, 1)
        for uRec_ in self.uRec_list:
            vfv.BuildURec_1(uRec_, 1)

        nFree = 0
        for iCell in range(mesh.NumCell()):
            if (
                np.linalg.norm(
                    vfv.GetCellBary(iCell).flatten() - np.array([0.5, 0.5, 0])
                )
                < 0.1
            ):
                self.iCellFree = iCell
                nFree += 1
        assert nFree == 1

    def test_one_wave(self, kx=0.1, ky=0.0, n_iter=10000, tol=1e-15):
        vfv = self.vfv
        mesh = self.mesh
        eval = self.eval

        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        uRec_real, uRecNew_real, uRec_imag, uRecNew_imag = self.uRec_list

        for iCell in range(self.mesh.NumCell()):
            u_i_real = np.array(u_real[iCell], copy=False)
            u_i_imag = np.array(u_imag[iCell], copy=False)
            xc = vfv.GetCellBary(iCell)
            xcr = xc - vfv.GetCellBary(self.iCellFree)
            wave_val = np.exp(1j * (kx * xcr[0] + ky * xcr[1]))
            u_i_real[0], u_i_imag[0] = np.real(wave_val), np.imag(wave_val)

            np.array(uRec_real[iCell], copy=False)[:] = 0.0
            np.array(uRec_imag[iCell], copy=False)[:] = 0.0

            # print(f"=== Cell {iCell}")
            # print(xcr)
            # print(wave_val)
            # print(u_real[iCell].tolist())
            # print(u_imag[iCell].tolist())

        for iter in range(1, 1 + n_iter):

            uRecPrevFree = np.array(
                (
                    np.array(uRec_real[self.iCellFree]),
                    np.array(uRec_imag[self.iCellFree]),
                )
            )
            eval.DoReconstructionIter(uRec_real, uRecNew_real, u_real, 0.0, False)
            eval.DoReconstructionIter(uRec_imag, uRecNew_imag, u_imag, 0.0, False)
            for iCell in range(self.mesh.NumCell()):
                if iCell == self.iCellFree:
                    continue
                xc = vfv.GetCellBary(iCell)
                xcr = xc - vfv.GetCellBary(self.iCellFree)
                wave_val = np.exp(1j * (kx * xcr[0] + ky * xcr[1]))
                # fmt: off
                np.array(uRec_real[iCell], copy=False)[:] = \
                    np.array(uRec_real[self.iCellFree]) * np.real(wave_val) - \
                    np.array(uRec_imag[self.iCellFree]) * np.imag(wave_val)
                np.array(uRec_imag[iCell], copy=False)[:] = \
                    np.array(uRec_imag[self.iCellFree]) * np.real(wave_val) + \
                    np.array(uRec_real[self.iCellFree]) * np.imag(wave_val)
                # fmt: on
            uRecNewFree = np.array(
                (
                    np.array(uRec_real[self.iCellFree]),
                    np.array(uRec_imag[self.iCellFree]),
                )
            )
            # print((uRecPrevFree - uRecNewFree).reshape((2, 9)))
            incNorm = np.linalg.norm((uRecPrevFree - uRecNewFree).flatten(), ord=1)

            if iter % 10 == 0:
                print(f"iter {iter} [{incNorm}]")
            if incNorm < tol:
                break

        eval.EvaluateRHS(rhs_real, u_real, uRec_real, 0.0)
        eval.EvaluateRHS(rhs_imag, u_imag, uRec_imag, 0.0)
        return (
            rhs_real[self.iCellFree].tolist()[0][0]
            + 1j * rhs_imag[self.iCellFree].tolist()[0][0]
        )


def test():
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()

    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    tester = WaveTester(mpi)
    print(tester.test_one_wave(np.pi * 0.1, 0.0))
    kxs = np.linspace(0, 1, 101) * np.pi
    kappaNum = np.zeros_like(kxs, dtype=np.complex128)
    for ikx, kx in enumerate(kxs):
        kappaNum[ikx] = 1j * tester.test_one_wave(kx, 0.0)
        



if __name__ == "__main__":
    test()
