import DNDSR.CFV._ext.cfv_pybind11
import DNDSR.DNDS._ext.dnds_pybind11
import DNDSR.Geom._ext.geom_pybind11
from __future__ import annotations
import collections.abc
import typing
from . import placeholder_submodule
__all__: list[str] = ['BC', 'BCHandler', 'BCInput', 'BCType', 'Evaluator', 'Physics', 'placeholder_submodule']
class BC:
    type: BCType
    def __init__(self) -> None:
        ...
    @property
    def id(self) -> int:
        ...
    @id.setter
    def id(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def values(self) -> list[float]:
        ...
    @values.setter
    def values(self, arg1: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
class BCHandler:
    def __init__(self, arg0: collections.abc.Sequence[BCInput], arg1: DNDSR.Geom._ext.geom_pybind11.AutoAppendName2ID) -> None:
        ...
    def id2bc(self, arg0: typing.SupportsInt) -> BC:
        ...
class BCInput:
    name: str
    type: BCType
    @staticmethod
    def from_dict(arg0: dict) -> BCInput:
        ...
    def __init__(self) -> None:
        ...
    def to_dict(self) -> dict:
        ...
    @property
    def value(self) -> list[float]:
        ...
    @value.setter
    def value(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
class BCType:
    """
    Members:
    
      Wall
    
      WallInvis
    
      WallIsothermal
    
      Far
    
      Sym
    
      In
    
      InPsTs
    
      Out
    
      OutP
    
      Special
    
      Unknown
    """
    Far: typing.ClassVar[BCType]  # value = <BCType.Far: 1>
    In: typing.ClassVar[BCType]  # value = <BCType.In: 7>
    InPsTs: typing.ClassVar[BCType]  # value = <BCType.InPsTs: 8>
    Out: typing.ClassVar[BCType]  # value = <BCType.Out: 5>
    OutP: typing.ClassVar[BCType]  # value = <BCType.OutP: 6>
    Special: typing.ClassVar[BCType]  # value = <BCType.Special: 10>
    Sym: typing.ClassVar[BCType]  # value = <BCType.Sym: 9>
    Unknown: typing.ClassVar[BCType]  # value = <BCType.Unknown: 0>
    Wall: typing.ClassVar[BCType]  # value = <BCType.Wall: 2>
    WallInvis: typing.ClassVar[BCType]  # value = <BCType.WallInvis: 3>
    WallIsothermal: typing.ClassVar[BCType]  # value = <BCType.WallIsothermal: 4>
    __members__: typing.ClassVar[dict[str, BCType]]  # value = {'Wall': <BCType.Wall: 2>, 'WallInvis': <BCType.WallInvis: 3>, 'WallIsothermal': <BCType.WallIsothermal: 4>, 'Far': <BCType.Far: 1>, 'Sym': <BCType.Sym: 9>, 'In': <BCType.In: 7>, 'InPsTs': <BCType.InPsTs: 8>, 'Out': <BCType.Out: 5>, 'OutP': <BCType.OutP: 6>, 'Special': <BCType.Special: 10>, 'Unknown': <BCType.Unknown: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Evaluator:
    class Cons2PrimMu_Arg:
        T: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        a: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        gamma: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        mu: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        p: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uGrad: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uGradPrim: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uPrim: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        def __init__(self) -> None:
            ...
        @property
        def muComp(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @muComp.setter
        def muComp(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalar.setter
        def uScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarGrad(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGrad.setter
        def uScalarGrad(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarGradPrim(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGradPrim.setter
        def uScalarGradPrim(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarPrim(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarPrim.setter
        def uScalarPrim(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
    class Cons2Prim_Arg:
        T: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        a: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        gamma: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        p: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uPrim: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        def __init__(self) -> None:
            ...
        @property
        def uScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalar.setter
        def uScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarPrim(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarPrim.setter
        def uScalarPrim(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
    class EstEigenDt_Arg:
        aCell: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        deltaLamCell: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        deltaLamFace: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        dt: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        faceLamEst: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1
        faceLamVisEst: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        muCell: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        def __init__(self) -> None:
            ...
    class Flux2nd_Arg:
        T: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        a: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        deltaLamCell: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        fluxFF: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        gamma: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        mu: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        p: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        pFL: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        pFR: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1
        rhs: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uFL: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uFR: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uGrad: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uGradFF: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uGradPrim: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uPrim: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        def __init__(self) -> None:
            ...
        @property
        def fluxScalarFF(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @fluxScalarFF.setter
        def fluxScalarFF(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def muComp(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @muComp.setter
        def muComp(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def rhsScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @rhsScalar.setter
        def rhsScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalar.setter
        def uScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarFL(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarFL.setter
        def uScalarFL(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarFR(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarFR.setter
        def uScalarFR(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarGrad(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGrad.setter
        def uScalarGrad(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarGradFF(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGradFF.setter
        def uScalarGradFF(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarGradPrim(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGradPrim.setter
        def uScalarGradPrim(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarPrim(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarPrim.setter
        def uScalarPrim(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
    class RecFace2nd_Arg:
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uFL: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uFR: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uGrad: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        uGradFF: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        def __init__(self) -> None:
            ...
        @property
        def uScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalar.setter
        def uScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarFL(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarFL.setter
        def uScalarFL(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarFR(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalarFR.setter
        def uScalarFR(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarGrad(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGrad.setter
        def uScalarGrad(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
        @property
        def uScalarGradFF(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGradFF.setter
        def uScalarGradFF(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
    class RecGradient_Arg:
        u: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1
        uGrad: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_5
        def __init__(self) -> None:
            ...
        @property
        def uScalar(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]:
            ...
        @uScalar.setter
        def uScalar(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1]) -> None:
            ...
        @property
        def uScalarGrad(self) -> list[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]:
            ...
        @uScalarGrad.setter
        def uScalarGrad(self, arg0: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1]) -> None:
            ...
    bcHandler: BCHandler
    fv: DNDSR.CFV._ext.cfv_pybind11.FiniteVolume
    physics: Physics
    def Cons2Prim(self, arg0: Evaluator.Cons2Prim_Arg) -> None:
        ...
    def Cons2PrimMu(self, arg0: Evaluator.Cons2PrimMu_Arg) -> None:
        ...
    def EstEigenDt(self, arg0: Evaluator.EstEigenDt_Arg) -> None:
        ...
    def Flux2nd(self, arg0: Evaluator.Flux2nd_Arg) -> None:
        ...
    def PrintDataVTKHDF(self, fname: str, series_name: str, arrCellCentScalar: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1] = [], arrCellCentScalar_names: collections.abc.Sequence[str] = [], arrCellCentVec: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1] = [], arrCellCentVec_names: collections.abc.Sequence[str] = [], arrNodeScalar: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_1_1] = [], arrNodeScalar_names: collections.abc.Sequence[str] = [], arrNodeVec: collections.abc.Sequence[DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_3_1] = [], arrNodeVec_names: collections.abc.Sequence[str] = [], uPrimCell: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1 = None, uPrimNode: DNDSR.DNDS._ext.dnds_pybind11.ArrayDOF_5_1 = None, t: typing.SupportsFloat = 0.0) -> None:
        ...
    def RecFace2nd(self, arg0: Evaluator.RecFace2nd_Arg) -> None:
        ...
    def RecGradient(self, arg0: Evaluator.RecGradient_Arg) -> None:
        ...
    def __init__(self, arg0: DNDSR.CFV._ext.cfv_pybind11.FiniteVolume, arg1: BCHandler, arg2: Physics) -> None:
        ...
    def device(self) -> DNDSR.DNDS._ext.dnds_pybind11.DeviceBackend:
        ...
    def getConfig(self) -> json:
        ...
    def setConfig(self, arg0: json) -> None:
        ...
    def to_device(self, arg0: str) -> None:
        ...
    def to_host(self) -> None:
        ...
class Physics:
    @staticmethod
    def from_dict(arg0: dict) -> Physics:
        ...
    def __init__(self) -> None:
        ...
    def to_dict(self) -> dict:
        ...
