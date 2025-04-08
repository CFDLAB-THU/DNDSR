import DNDS
import Geom
import sys, os, math
import mpi4py.MPI as MPI
import numpy as np
from utils import *


class OversetPart2D:
    def __init__(self, mpi: DNDS.MPIInfo, transform=(np.eye(3, 3), np.zeros(3))):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)
        self._transform = transform
        assert transform[0].shape == (3, 3)
        assert transform[1].shape == (3,)

    def coord_mesh_to_phy(self, coord_mesh):
        if coord_mesh.ndim == 2:
            return self._transform[0] @ coord_mesh + self._transform[1][:, np.newaxis]
        elif coord_mesh.ndim == 1:
            return self._transform[0] @ coord_mesh + self._transform[1]

    def read_mesh(self, f_name: str):
        self._mesh, self._meshReader, self._name2IDAutoAppend = get_mesh_2D(
            f_name, self._mpi
        )
        self._name2ID = self._name2IDAutoAppend.n2id_map
        self._id2name = {}
        for name, id in self._name2ID.items():
            self._id2name[id] = name
        coordsFatherData = np.array(self._mesh.coords.father.data())
        # print(coordsFatherData.shape)
        coordsFatherData = coordsFatherData.reshape((3, -1), order="F")
        coordsFatherData = self.coord_mesh_to_phy(coordsFatherData)
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

    def obtain_dist_node(self):
        from OversetCartUtil import obtain_part_local_elem_dists

        self.dist_node = obtain_part_local_elem_dists(self)


class OversetBG2D:
    def __init__(self, mpi: DNDS.MPIInfo):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)

    def set_bg(self, parts: list[OversetPart2D], h: float):
        self.h = h
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

    def rank_to_ax_rank(self, rank=None):
        if rank is None:
            rank = self._mpi.rank
        return (rank // self.nProcAx[1], rank % self.nProcAx[1])

    def proc_cell_grid_shape(self):
        ax_rank = self.rank_to_ax_rank()
        return tuple(
            [
                self.nStarts[i][ax_rank[i] + 1] - self.nStarts[i][ax_rank[i]]
                for i in range(2)
            ]
        )

    def proc_point_grid_shape(self):
        ax_rank = self.rank_to_ax_rank()
        return tuple(
            [
                self.nStarts4point[i][ax_rank[i] + 1]
                - self.nStarts4point[i][ax_rank[i]]
                for i in range(2)
            ]
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

    def obtain_dist_map(self, parts: list[OversetPart2D]):

        MPI = self._MPI
        mpi = self._mpi
        from OversetCartUtil import obtain_part_local_dists

        for part in parts:
            part_local_dists = obtain_part_local_dists(self, part)
            part_local_dists_items = list(part_local_dists.items())

            ijkv = np.array([v[0] for v in part_local_dists_items]).transpose()
            if ijkv.size:
                ranks = list(self.ijk_to_rank(ijkv))
            else:
                ranks = []
            # sendLists = [[]] * self._mpi.size # ! this is wrong!!! the empty lists of each item refs the same
            sendLists = [[] for _ in range(mpi.size)]

            for i, rank in enumerate(ranks):
                datac = part_local_dists_items[i]
                data = ((int(datac[0][0]), int(datac[0][1])), float(datac[1]))
                sendLists[rank].append(datac)
                ax_r = self.rank_to_ax_rank(rank)

            recvLists = MPI.alltoall(sendLists)
            # print(f"{mpi.rank}, {([len(v) for v in sendLists])}, {len(sendLists)}")
            # return

            ax_ranks = self.rank_to_ax_rank()
            proc_bg_mesh_dist = np.ones(self.proc_point_grid_shape()) * 1e300
            for recvL in recvLists:
                # print(recvL)
                for g_point, v in recvL:
                    for ax in range(2):
                        assert (
                            g_point[ax] < self.nStarts4point[ax][ax_ranks[ax] + 1]
                        ), g_point[ax]
                    l_point = (
                        g_point[0] - self.nStarts4point[0][ax_ranks[0]],
                        g_point[1] - self.nStarts4point[1][ax_ranks[1]],
                    )
                    proc_bg_mesh_dist[l_point] = v
            print(f"proc {mpi.rank}: min dist at grid point: {proc_bg_mesh_dist.min()}")


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
    osPart.obtain_dist_node()

    osBG = OversetBG2D(mpi)
    osBG.set_bg([osPart], 1.0 / 2)
    assert osBG.procMap[osBG.rank_to_ax_rank()] == mpi.rank
    osBG.obtain_dist_map([osPart])

    if mpi.rank == 0:
        _ = osBG.ijk_to_rank(
            np.random.randint(0, min(osBG.grid_shape[0:2]) - 1, size=(2, 32))
        )

        print(osBG.ijk_to_rank([v - 1 for v in osBG.grid_shape], is_point=True))

        try:
            print(osBG.ijk_to_rank([v - 1 for v in osBG.grid_shape], is_point=False))
        except ValueError as e:
            print(e)
            print("caught")
        else:
            raise RuntimeError("not raising value error")
