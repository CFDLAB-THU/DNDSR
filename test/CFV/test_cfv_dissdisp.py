import DNDSR

# print(DNDSR.__dict__)

from DNDSR import DNDS, Geom, CFV
from DNDSR.Geom.utils import *
import numpy as np
import json

# vfvSettings = json.loads(
#     "".join(
#         [
#             line if not line.strip().startswith("//") else ""
#             for line in """{
#     "maxOrder": 3,
#     "intOrder": 5,
#     "intOrderVR": 5,
#     "cacheDiffBase": true,
#     "jacobiRelax": 1,
#     "SORInstead": false,
#     "smoothThreshold": 1e-3,
#     "WBAP_nStd": 10.0,
#     "normWBAP": false,
#     "subs2ndOrder": 1,
#     "subs2ndOrderGGScheme": 0,
#     "baseSettings": {
#         "localOrientation": false,
#         "anisotropicLengths": false
#     },
#     "functionalSettings": {
#         // "scaleType": "MeanAACBB",
#         "dirWeightScheme": "HQM_OPT",
#         // "dirWeightScheme": "ManualDirWeight",
#         // "manualDirWeights": [
#         //     1.0,
#         //     1,
#         //     0,
#         //     0
#         // ],
#         "geomWeightScheme": "HQM_SD",
#         "geomWeightPower": 0.5,
#         "geomWeightBias": 1,
#         // "geomWeightScheme": "SD_Power",
#         // "geomWeightPower1": -0.5,
#         // "geomWeightPower2": 0.5,
#         // "useAnisotropicFunctional": true,
#         // // "anisotropicType": "InertiaCoordBB",
#         // "inertiaWeightPower": 0,
#         // "scaleMultiplier": 1,
#         "_tail": 0
#     }
# }
# """.splitlines()
#         ]
#     )
# )
# print(vfvSettings)


def default_VRSettings():
    return {
        "maxOrder": 3,
        "intOrder": 5,
        "intOrderVR": 5,
        "cacheDiffBase": True,
        "jacobiRelax": 1,
        "SORInstead": False,
        "smoothThreshold": 0.001,
        "WBAP_nStd": 10.0,
        "normWBAP": False,
        "subs2ndOrder": 1,
        "subs2ndOrderGGScheme": 0,
        "baseSettings": {"localOrientation": False, "anisotropicLengths": False},
        "functionalSettings": {
            "dirWeightScheme": "HQM_OPT",
            "geomWeightScheme": "HQM_SD",
            "geomWeightPower": 0.5,
            "geomWeightBias": 1,
            "_tail": 0,
        },
    }


class WaveTester:

    def __init__(self, mpi, vfvSettings=default_VRSettings(), ax=1.0, ay=0.0):
        self.mpi = mpi
        self.ax = ax
        self.ay = ay

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

        self.reader = reader
        self.name2Id = name2Id
        self.meshBnd = meshBnd
        self.readerBnd = readerBnd

        vfv = CFV.VariationalReconstruction_2(mpi, mesh)
        self.vfvSettings = vfvSettings
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
        self.eval = CFV.ModelEvaluator(mesh, vfv, {"ax": ax, "ay": ay})

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

    def update_vfv_settings(self, vfvSettings):
        mpi = self.mpi
        vfv = self.vfv
        mesh = self.mesh
        eval = self.eval
        name2Id = self.name2Id
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

    def get_rhsFreeComplex(self):
        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        return np.array(rhs_real[self.iCellFree]) + 1j * np.array(
            rhs_imag[self.iCellFree]
        )

    def get_uFreeComplex(self):
        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        return np.array(u_real[self.iCellFree]) + 1j * np.array(u_imag[self.iCellFree])

    def set_uFree(self, r: float, i: float):
        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        np.array(u_real[self.iCellFree], copy=False)[:] = r
        np.array(u_imag[self.iCellFree], copy=False)[:] = i

    def test_one_wave(
        self,
        kx=0.1,
        ky=0.0,
        n_iter=10000,
        tol=1e-15,
        n_print=0,
        n_iter_min=1,
    ):
        vfv = self.vfv
        mesh = self.mesh
        eval = self.eval

        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        uRec_real, uRecNew_real, uRec_imag, uRecNew_imag = self.uRec_list

        self.set_uFree(1, 0)
        self.uSync(kx, ky)

        self.DoReconstruction(kx, ky, n_iter, tol, n_print, n_iter_min=n_iter_min)

        eval.EvaluateRHS(rhs_real, u_real, uRec_real, 0.0)
        eval.EvaluateRHS(rhs_imag, u_imag, uRec_imag, 0.0)
        return (
            rhs_real[self.iCellFree].tolist()[0][0]
            + 1j * rhs_imag[self.iCellFree].tolist()[0][0]
        )

    def test_conv_rate(
        self,
        dTau=10,
        kx=0.1,
        ky=0.0,
        n_iter=10000,
        tol=1e-15,
        n_print=0,
        n_iter_min=1,
        singlegrid_niter=1,
        multigrid_niters=(0, 0),
    ):
        vfv = self.vfv
        mesh = self.mesh
        eval = self.eval

        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        uRec_real, uRecNew_real, uRec_imag, uRecNew_imag = self.uRec_list

        # build the fv jacobian
        J = np.complex128(0)
        c2f = np.array(mesh.cell2face[self.iCellFree])
        xcC = vfv.GetCellBary(self.iCellFree)

        for ic2f, iFace in enumerate(c2f):
            iCellOther = mesh.CellFaceOther(self.iCellFree, iFace)
            normOut = vfv.GetFaceNormFromCell(iFace, self.iCellFree, -1, -1)
            if not mesh.CellIsFaceBack(self.iCellFree, iFace):
                normOut *= -1.0
            a_out = normOut[0] * self.ax + normOut[1] * self.ay

            xcr = vfv.GetCellBary(iCellOther) - xcC
            wave_val = np.exp(1j * (kx * xcr[0] + ky * xcr[1]))
            dFdu = (1 + wave_val) * 0.5 * a_out - 0.5 * np.abs(a_out) * (wave_val - 1)
            J -= dFdu * vfv.GetFaceArea(iFace) / vfv.GetCellVol(self.iCellFree)

        np.array(u_real[self.iCellFree], copy=False)[:] = 1
        np.array(u_imag[self.iCellFree], copy=False)[:] = 0
        self.uSync(kx, ky)

        for iter in range(singlegrid_niter):
            self.DoReconstruction(kx, ky, n_iter, tol, n_print, n_iter_min=n_iter_min)
            eval.EvaluateRHS(rhs_real, u_real, uRec_real, 0.0)
            eval.EvaluateRHS(rhs_imag, u_imag, uRec_imag, 0.0)

            uFreeNew = self.get_uFreeComplex() + self.get_rhsFreeComplex() / (
                -J + 1.0 / dTau
            )
            self.set_uFree(np.real(uFreeNew), np.imag(uFreeNew))
            self.uSync(kx, ky)

        for iter in range(multigrid_niters[0]):
            # not need reconstruction
            # self.DoReconstruction(kx, ky, n_iter, tol, n_print, n_iter_min=n_iter_min)
            options = eval.EvaluateRHSOptions()
            options.direct2ndRec = True
            options.direct2ndRec1stConv = False
            eval.EvaluateRHS(rhs_real, u_real, uRec_real, 0.0, options=options)
            eval.EvaluateRHS(rhs_imag, u_imag, uRec_imag, 0.0, options=options)

            uFreeNew = self.get_uFreeComplex() + self.get_rhsFreeComplex() / (
                -J + 1.0 / dTau
            )
            self.set_uFree(np.real(uFreeNew), np.imag(uFreeNew))
            self.uSync(kx, ky)

        for iter in range(multigrid_niters[1]):
            # not need reconstruction
            # self.DoReconstruction(kx, ky, n_iter, tol, n_print, n_iter_min=n_iter_min)
            options = eval.EvaluateRHSOptions()
            options.direct2ndRec = True
            options.direct2ndRec1stConv = True
            eval.EvaluateRHS(rhs_real, u_real, uRec_real, 0.0, options=options)
            eval.EvaluateRHS(rhs_imag, u_imag, uRec_imag, 0.0, options=options)

            uFreeNew = self.get_uFreeComplex() + self.get_rhsFreeComplex() / (
                -J + 1.0 / dTau
            )
            self.set_uFree(np.real(uFreeNew), np.imag(uFreeNew))
            self.uSync(kx, ky)

        return self.get_uFreeComplex().flatten()[0]

    def uRecSync(self, kx: float, ky: float):
        vfv = self.vfv
        uRec_real, uRecNew_real, uRec_imag, uRecNew_imag = self.uRec_list
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

    def uSync(self, kx: float, ky: float):
        vfv = self.vfv
        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        for iCell in range(self.mesh.NumCell()):
            if iCell == self.iCellFree:
                continue
            xc = vfv.GetCellBary(iCell)
            xcr = xc - vfv.GetCellBary(self.iCellFree)
            wave_val = np.exp(1j * (kx * xcr[0] + ky * xcr[1]))
            # fmt: off
            np.array(u_real[iCell], copy=False)[:] = \
                np.array(u_real[self.iCellFree]) * np.real(wave_val) - \
                np.array(u_imag[self.iCellFree]) * np.imag(wave_val)
            np.array(u_imag[iCell], copy=False)[:] = \
                np.array(u_imag[self.iCellFree]) * np.real(wave_val) + \
                np.array(u_real[self.iCellFree]) * np.imag(wave_val)
            # fmt: on

    def DoReconstruction(
        self, kx, ky, n_iter=10000, tol=1e-15, n_print=0, n_iter_min=1
    ):
        vfv = self.vfv
        mesh = self.mesh
        eval = self.eval
        u_real, rhs_real, u_imag, rhs_imag = self.u_list
        uRec_real, uRecNew_real, uRec_imag, uRecNew_imag = self.uRec_list

        if self.vfvSettings["subs2ndOrder"] == 0 or self.vfvSettings["maxOrder"] > 1:
            # use matrices:
            c2f = np.array(mesh.cell2face[self.iCellFree])
            matrixAAInvB = vfv.matrixAAInvB
            vectorAInvB = vfv.vectorAInvB
            AInv = np.array(matrixAAInvB[self.iCellFree, 0])
            M, N = AInv.shape
            assert M == N
            MatC = np.eye(M, M, dtype=np.complex128)
            RhsC = np.zeros((M, 1), dtype=np.complex128)
            xcC = vfv.GetCellBary(self.iCellFree)
            uC = np.array(u_real[self.iCellFree]) + 1j * np.array(
                u_imag[self.iCellFree]
            )
            for ic2f, iFace in enumerate(c2f):
                iCellOther = mesh.CellFaceOther(self.iCellFree, iFace)
                xcr = vfv.GetCellBary(iCellOther) - xcC
                wave_val = np.exp(1j * (kx * xcr[0] + ky * xcr[1]))
                # print(np.array(matrixAAInvB[self.iCellFree, ic2f + 2], copy=False))
                MatC -= wave_val * np.array(
                    matrixAAInvB[self.iCellFree, ic2f + 1], copy=False
                )
                uOther = np.array(u_real[iCellOther]) + 1j * np.array(
                    u_imag[iCellOther]
                )
                RhsC += (uOther - uC) * np.array(
                    vectorAInvB[self.iCellFree, ic2f], copy=False
                )
            # print(np.linalg.eig(MatC))
            uRecFree = np.linalg.solve(MatC, RhsC)
            np.array(uRec_real[self.iCellFree], copy=False)[:] = np.real(uRecFree)
            np.array(uRec_imag[self.iCellFree], copy=False)[:] = np.imag(uRecFree)
            self.uRecSync(kx, ky)
            return

        for iCell in range(self.mesh.NumCell()):
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
            self.uRecSync(kx, ky)
            uRecNewFree = np.array(
                (
                    np.array(uRec_real[self.iCellFree]),
                    np.array(uRec_imag[self.iCellFree]),
                )
            )
            # print((uRecPrevFree - uRecNewFree).reshape((2, 9)))
            incNorm = np.linalg.norm((uRecPrevFree - uRecNewFree).flatten(), ord=1)

            if iter >= n_iter_min and n_print > 0 and iter % n_print == 0:
                print(f"iter {iter} [{incNorm}]")
            if incNorm < tol:
                break

        # print(np.array(uRec_real[self.iCellFree]).flatten())
        # print(np.real(uRecFree).flatten())


def test():
    print(CFV.tUDof_1)
    # CFV.VariationalReconstruction_2()

    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    tester = WaveTester(mpi)
    print(tester.test_one_wave(np.pi * 0.1, 0.0))

    # kxs = np.linspace(0, 1, 101) * np.pi
    # kappaNum = np.zeros_like(kxs, dtype=np.complex128)
    # for ikx, kx in enumerate(kxs):
    #     kappaNum[ikx] = 1j * tester.test_one_wave(kx, 0.0)

    print(tester.test_conv_rate(10, kx=np.pi * 0.1, ky=0, multigrid_niters=(4, 0)))


if __name__ == "__main__":
    test()
