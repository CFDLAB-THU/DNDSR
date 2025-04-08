import DNDS
import Geom
import sys, os, math
import mpi4py.MPI as MPI
import numpy as np
from utils import *


class OversetPart2D:
    def __init__(self, mpi: DNDS.MPIInfo):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)

    def read_mesh(self, f_name: str):
        self._mesh, self._meshReader, self._name2ID = get_mesh_2D(f_name, self._mpi)
        coordsFatherData = np.array(self._mesh.coords.father.data())
        # print(coordsFatherData.shape)
        coordsFatherData = coordsFatherData.reshape((3, -1), order="F")
        coordsMax = coordsFatherData.max(axis=1)
        coordsMin = coordsFatherData.min(axis=1)
        self._MPI.Allreduce(None, coordsMax, op=MPI.MAX)
        self._MPI.Allreduce(None, coordsMin, op=MPI.MIN)
        self._xyzMax = coordsMax
        self._xyzMin = coordsMin

    @property
    def xyzMinMax(self):
        return np.concatenate(
            (self._xyzMin.transpose(), self._xyzMax.transpose())
        ).reshape((2, 3), order="C")


class OversetBG2D:
    def __init__(self, mpi: DNDS.MPIInfo):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)

    def set_bg(self, parts: list[OversetPart2D], h: float):
        box = np.zeros((2, 3))
        box[0, :] = 1e300
        box[1, :] = -1e300
        for part in parts:
            partBox = part.xyzMinMax
            box[0, :] = np.minimum(partBox[0, :], box[0, :])
            box[1, :] = np.maximum(partBox[1, :], box[1, :])
        xNodes = []
        origins = np.zeros(3) * np.nan
        for iax in range(3):
            lower = box[0, iax]
            upper = box[1, iax]
            lower -= h / 2
            upper += h / 2
            nInterval = int(np.ceil((upper - lower) / h))
            assert nInterval >= 0
            upperV = lower + nInterval * h
            assert upperV > box[1, iax]
            xNodes.append(np.linspace(lower, upperV, nInterval + 1, dtype=np.float64))
            origins[iax] = lower

        self.xNodes = xNodes
        self.grid_shape = tuple([v.size for v in xNodes])
        self.origins = origins

        self.grid_shape = self._MPI.bcast(self.grid_shape, root=0)
        self.origins = self._MPI.bcast(self.origins, root=0)
        if self._mpi.rank == 0:

            # print(self.grid_shape)
            # print(self.origins)

            # do BG grid partition
            from ProductDecompose import int_factor_divide

            nProcAx = int_factor_divide(self._mpi.size, 2)  # 2d
            nIntervals = [v - 1 for v in self.grid_shape]
            nStarts = []
            for iax in range(2):
                nIntervalP = int(np.floor(nIntervals[iax] / nProcAx[iax]))
                nIntervalPLast = nIntervals[iax] - (nProcAx[iax] - 1) * nIntervalP
                assert nIntervalPLast >= 0
                nStartsAx = [i * nIntervalP for i in range(nProcAx[iax])] + [
                    nIntervals[iax]
                ]
                assert nStartsAx[-1] - nStartsAx[-2] == nIntervalPLast
                nStarts.append(np.array(nStartsAx, dtype=np.int64))
            # print(nStarts)
            self.nStarts = nStarts
            self.nProcAx = nProcAx
        self.nStarts, self.nProcAx = self._MPI.bcast(
            (self.nStarts, self.nProcAx) if self._mpi.rank == 0 else None, root=0
        )
        self.procMap = np.arange(self._mpi.size, dtype=np.int64).reshape(
            self.nProcAx, order="C"
        )
        self.nStarts4point = [np.copy(ar) for ar in self.nStarts]
        for v in self.nStarts4point:
            v[-1] += 1
        if self._mpi.rank == 0:
            print(self.procMap)
            print(
                f"{self._mpi.rank} nStarts: {self.nStarts4point}, nProcAx: {self.nProcAx}"
            )

    def ijk_to_rank(self, ijk, is_point=False):
        idxs = []
        for iax in range(2):
            # searchsorted gives the insertion point for target
            indexAx = (
                np.searchsorted(
                    self.nStarts4point[iax] if is_point else self.nStarts[iax],
                    ijk[iax],
                    side="right",
                )
                - 1
            )
            if np.any(indexAx < 0) or np.any(indexAx >= self.nProcAx[iax]):
                raise ValueError(f"ijk {ijk} out of range")
            idxs.append(indexAx)
        # print(idxs)
        return self.procMap[tuple(idxs)]


if __name__ == "__main__":
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    osPart = OversetPart2D(mpi)
    osPart.read_mesh(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "mesh",
            "NACA0012_H2.cgns",
        )
    )

    osBG = OversetBG2D(mpi)
    osBG.set_bg([osPart], 1.0 / 10)

    if mpi.rank == 0:
        print(
            osBG.ijk_to_rank(
                np.random.randint(0, min(osBG.grid_shape[0:2]) - 1, size=(2, 32))
            )
        )

        print(osBG.ijk_to_rank([v - 1 for v in osBG.grid_shape], is_point=True))

        try:
            print(osBG.ijk_to_rank([v - 1 for v in osBG.grid_shape], is_point=False))
        except ValueError as e:
            print(e)
            print("caught")
        else:
            raise RuntimeError("not raising value error")
