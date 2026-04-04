from DNDSR import DNDS, Geom, CFV, EulerP
import inspect
import json
from DNDSR.Geom.utils import *
import numpy as np
import pprint
import math
import copy


class Solver:
    _runningDevice = DNDS.DeviceBackend.Unknown

    def __init__(self, mpi: DNDS.MPIInfo):
        self.mpi = mpi
        pass

    @property
    def runningDevice(self):
        return self._runningDevice

    @runningDevice.setter
    def runningDevice(self, B):
        self._runningDevice = B

    def ReadMesh(self, mesh_file, dim, other_options={}):
        self.mesh, self.reader, self.name2Id = create_mesh_from_CGNS(
            mesh_file,
            mpi=self.mpi,
            dim=dim,
            **other_options,
        )
        self.meshBnd, self.readerBnd = create_bnd_mesh(self.mesh)

    def BuildFV(self, fv_settings):
        mpi = self.mpi
        mesh = self.mesh
        fv = CFV.FiniteVolume(mpi, mesh)
        if mpi.rank == 0:
            print(fv.GetSettings())
        settings = fv.GetSettings()
        settings.update(fv_settings)
        if mpi.rank == 0:
            print(f"FV Settings: \n{settings}")
        fv.ParseSettings(settings)
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

        self.fv = fv
        nB = fv.getArrayBytes() / 1024**2
        nBMesh = mesh.getArrayBytes() / 1024**2
        if mpi.rank == 0:
            print(f"Bytes     : {nB:.4g} MB")
            print(f"Bytes mesh: {nBMesh:.4g} MB")

    def BuildEval(self):
        mpi = self.mpi
        mesh, reader, name2Id, meshBnd, readerBnd, fv = (
            self.mesh,
            self.reader,
            self.name2Id,
            self.meshBnd,
            self.readerBnd,
            self.fv,
        )
        if mpi.rank == 0:
            print("Name 2 Id Map: ")
            pprint.pprint(name2Id.n2id_map)

        phys_dict = EulerP.Physics().to_dict()
        phys_params = phys_dict["params"]
        phys_params["TRef"] = 273.15
        phys_params["cp"] = 1.0
        phys_params["gamma"] = 1.4
        phys_params["mu0"] = 1e-10
        phys_dict["reference_values"] = [1.0, 1.0, 1.0, 1.0, 1.0]

        phys = EulerP.Physics.from_dict(phys_dict)

        if mpi.rank == 0:
            print("Physics:")
            pprint.pprint(phys.to_dict())

        bcInputs = []
        bcHandler = EulerP.BCHandler(bcInputs, name2Id)

        eval = EulerP.Evaluator(fv, bcHandler, phys)

        self.eval = eval
        self.bcHandler = bcHandler
        self.phys = phys

    def WrapEval(self):
        # class EvalWrapper:
        #     def __init__(self, obj):
        #         self._obj = obj  # object being wrapped

        #     def __getattr__(self, name):
        #         # only called if attribute is not found on the Wrapper instance
        #         return getattr(self._obj, name)

        #     def __setattr__(self, name, value):
        #         if name == "_obj":
        #             super().__setattr__(name, value)
        #         else:
        #             setattr(self._obj, name, value)

        from DNDSR.DNDS.Wrapper import generate_wrapper

        Wrapper = generate_wrapper(type(self.eval), init_from_obj=True)

        self.eval = Wrapper(self.eval)

    def BuildDataArray(self):
        # TODO: handle scalars
        mpi = self.mpi
        mesh, reader, name2Id, meshBnd, readerBnd, fv = (
            self.mesh,
            self.reader,
            self.name2Id,
            self.meshBnd,
            self.readerBnd,
            self.fv,
        )

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

        self.u = u
        self.s = s
        self.gU = gU
        self.gs = gS
        self.uf = uf
        self.sf = sf
        self.gUf = gUf
        self.gsf = gSf

        data = {}

        data["u"] = u
        data["u0"] = u.clone()
        data["rhs"] = u.clone()
        data["rhs1"] = u.clone()
        data["rhs2"] = u.clone()

        data["uGrad"] = gU

        data["p"] = s.clone()
        data["T"] = s.clone()
        data["a"] = s.clone()
        data["mu"] = s.clone()
        data["gamma"] = s.clone()

        data["uPrim"] = u.clone()
        data["uGradPrim"] = gU.clone()

        data["deltaLamCell"] = s.clone()
        data["dt"] = s.clone()
        data["uFL"] = uf
        data["uGradFF"] = gUf
        data["deltaLamFace"] = sf.clone()
        data["faceLamEst"] = gSf.clone()
        data["faceLamVisEst"] = sf.clone()

        data["uGradFF"] = gUf.clone()
        data["uFL"] = uf.clone()
        data["uFR"] = uf.clone()

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

        data["fluxFF"] = uf.clone()

        self.data = data

    def data_to_device(self, backend=None):
        if backend is None:
            return
        for n, a in self.data.items():
            if isinstance(a, list):
                for aa in a:
                    aa.to_device(backend)
            else:
                a.to_device(backend)

    def WrapData(self):
        from DNDSR.DNDS.Wrapper import generate_wrapper

        class _WrappedView:
            def __init__(self, dict):
                self._dict = dict

            def __getitem__(self, key):
                return self._dict.get_wrapped(key)

        class WrappedDict(dict):
            def __init__(self):
                super().__init__()

            def __getitem__(self, key):
                return super().__getitem__(key)[0]

            def __setitem__(self, key, val):
                if isinstance(val, list):
                    val_wrapped = []
                    for i in range(len(val)):
                        val_wrapped.append(generate_wrapper(type(val[i]), True)(val[i]))
                else:  # TODO check types here
                    val_wrapped = generate_wrapper(type(val), True)(val)
                return super().__setitem__(key, (val, val_wrapped))

            def get_wrapped(self, key):
                return super().__getitem__(key)[1]

            def items(self):
                for k, v in super().items():
                    yield k, v[0]

            def Wrapped(self):
                return _WrappedView(self)

        data = self.data
        self.data = WrappedDict()
        self.data.update(data)

        for n, a in data.items():
            self.data[n] = a

    def to_device(self, backend=None):
        if backend is None:
            return
        if not hasattr(self, "mesh"):
            raise ValueError("need to initialize mesh before to_device")
        if not hasattr(self, "fv"):
            raise ValueError("need to initialize fv before to_device")
        if not hasattr(self, "eval"):
            raise ValueError("need to initialize eval before to_device")
        self.mesh.to_device(backend)
        self.fv.to_device(backend)
        self.eval.to_device(backend)

    def LoadArg_RecGradient(self):
        data = self.data
        eval = self.eval
        arg = eval.RecGradient_Arg()
        arg.u = data["u"]
        arg.uGrad = data["uGrad"]
        arg.uScalar = []
        arg.uScalarGrad = []
        return arg

    def LoadArg_Cons2PrimMu(self):
        data = self.data
        eval = self.eval
        arg = eval.Cons2PrimMu_Arg()
        arg.u = data["u"]
        arg.uGrad = data["uGrad"]
        arg.uPrim = data["uPrim"]
        arg.uGradPrim = data["uGradPrim"]

        arg.p = data["p"]
        arg.T = data["T"]
        arg.a = data["a"]
        arg.mu = data["mu"]
        arg.gamma = data["gamma"]

        return arg

    def LoadArg_EstEigenDt(self):
        data = self.data
        eval = self.eval
        arg = eval.EstEigenDt_Arg()
        arg.u = data["u"]
        arg.muCell = data["mu"]
        arg.aCell = data["a"]
        arg.deltaLamCell = data["deltaLamCell"]
        arg.deltaLamFace = data["deltaLamFace"]
        arg.faceLamEst = data["faceLamEst"]
        arg.faceLamVisEst = data["faceLamVisEst"]
        arg.dt = data["dt"]
        return arg

    def LoadArg_RecFace2nd(self):
        data = self.data
        eval = self.eval
        arg = eval.RecFace2nd_Arg()
        arg.u = data["u"]
        arg.uGrad = data["uGrad"]
        arg.uFL = data["uFL"]
        arg.uFR = data["uFR"]
        arg.uGradFF = data["uGradFF"]
        return arg

    def LoadArg_Cons2Prim_FX(self, FX: str = "FL"):
        data = self.data
        eval = self.eval
        if FX not in {"FL", "FR"}:
            raise ValueError(f"FX {FX} not supported")
        arg = eval.Cons2Prim_Arg()
        arg.u = data["u" + FX]
        arg.uPrim = data["uPrim" + FX]
        arg.p = data["p" + FX]
        arg.T = data["T" + FX]
        arg.a = data["a" + FX]
        arg.gamma = data["gamma" + FX]
        return arg

    def LoadArg_Flux2nd(self):
        data = self.data
        eval = self.eval
        arg = eval.Flux2nd_Arg()
        arg.u = data["u"]
        arg.uGrad = data["uGrad"]
        arg.uPrim = data["uPrim"]
        arg.uGradPrim = data["uGradPrim"]

        arg.p = data["p"]
        arg.T = data["T"]
        arg.a = data["a"]
        arg.gamma = data["gamma"]
        arg.mu = data["mu"]
        arg.deltaLamCell = data["deltaLamCell"]

        arg.uFL = data["uFL"]
        arg.uFR = data["uFR"]
        arg.pFL = data["pFL"]
        arg.pFR = data["pFR"]

        arg.uGradFF = data["uGradFF"]

        arg.fluxFF = data["fluxFF"]
        arg.rhs = data["rhs"]
        return arg

    def CalculateOneRHS(
        self,
        zero_grad=False,
    ):
        solver = self
        eval = self.eval
        mpi = self.mpi

        # data = self.data
        data = (
            self.data.Wrapped()
        )  #!caution, the wrapped objects cannot be put into pybind11 fields
        eval.RecGradient(solver.LoadArg_RecGradient())
        # print(data["p"].min())
        # _ = input()
        data["uGrad"].trans.startPersistentPull(self.runningDevice)
        if zero_grad:
            data["uGrad"].setConstant(0.0)
        eval.Cons2PrimMu(solver.LoadArg_Cons2PrimMu())
        data["p"].trans.startPersistentPull(self.runningDevice)
        data["T"].trans.startPersistentPull(self.runningDevice)
        data["a"].trans.startPersistentPull(self.runningDevice)
        data["gamma"].trans.startPersistentPull(self.runningDevice)
        data["mu"].trans.startPersistentPull(self.runningDevice)

        # TODO: C++ side check PPCheckCons()
        pMin = data["p"].min()
        aSum = data["a"].sum()
        if pMin <= 0.0 or not math.isfinite(pMin) or not math.isfinite(aSum):
            raise RuntimeError(f"pMin is {pMin}, aSum is {aSum}, pp Failed")
        eval.EstEigenDt(solver.LoadArg_EstEigenDt())
        data["deltaLamCell"].trans.startPersistentPull(self.runningDevice)
        eval.RecFace2nd(solver.LoadArg_RecFace2nd())
        eval.Cons2Prim(solver.LoadArg_Cons2Prim_FX("FL"))
        eval.Cons2Prim(solver.LoadArg_Cons2Prim_FX("FR"))
        pMin = min(data["pFL"].min(), data["pFR"].min())
        aSum = data["aFL"].sum() + data["aFR"].sum()
        if pMin <= 0.0 or not math.isfinite(pMin) or not math.isfinite(aSum):
            raise RuntimeError(f"pMin is {pMin}, aSum is {aSum}, pp Failed")
        data["rhs"].setConstant(0.0)
        eval.Flux2nd(solver.LoadArg_Flux2nd())

    def IntegrateDt_ExplicitInterval(
        self, tStart: float, tEnd: float, CFL: float, step0, max_step: int = 100000
    ):
        solver = self
        eval = self.eval
        mpi = self.mpi

        t = tStart
        data = self.data

        zero_grad = False

        for iStep in range(step0 + 1, max_step + 1):

            self.CalculateOneRHS(zero_grad=zero_grad)
            if_stop = False
            dt = data["dt"].min() * CFL
            tNext = t + dt
            if tNext >= tEnd:
                dt = tEnd - t
                tNext = tEnd
                if_stop = True

            data["u0"].assign_value(data["u"])
            data["rhs1"], data["rhs"] = data["rhs"], data["rhs1"]
            data["u"].addTo(data["rhs1"], dt)
            data["u"].trans.startPersistentPull(self.runningDevice)

            self.CalculateOneRHS(zero_grad=zero_grad)

            data["rhs2"], data["rhs"] = data["rhs"], data["rhs2"]
            data["u"].assign_value(data["u0"])
            data["u"].addTo(data["rhs1"], dt * 0.25)
            data["u"].addTo(data["rhs2"], dt * 0.25)
            data["u"].trans.startPersistentPull(self.runningDevice)

            self.CalculateOneRHS(zero_grad=zero_grad)
            data["u"].assign_value(data["u0"])
            data["u"].addTo(data["rhs1"], dt * 1.0 / 6)
            data["u"].addTo(data["rhs2"], dt * 1.0 / 6)
            data["u"].addTo(data["rhs"], dt * 4.0 / 6)
            data["u"].trans.startPersistentPull(self.runningDevice)

            t = tNext

            if iStep % 1 == 0 or if_stop:
                rhsNorm = data["rhs1"].componentWiseNorm1()
                pMax = data["p"].max()
                pMin = data["p"].min()
                uGradN = data["uGrad"].norm2()
                if mpi.rank == 0:
                    print(
                        f"Step [{iStep}], t [{t:.4e}] dt [{dt:.4e}] pRange [{pMax:.2e},{pMin:.2e}]"
                        + f" rhs: [{','.join([f'{v:8.2e}' for v in rhsNorm.tolist()])}]"
                    )
                    print(uGradN)

            if if_stop:
                return iStep, t
        return max_step, t
