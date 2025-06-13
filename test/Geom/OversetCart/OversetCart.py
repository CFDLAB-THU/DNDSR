import DNDS
import Geom
import sys, os, math
import mpi4py.MPI as pyMPI
import numpy as np
from utils import *
from CartGridField import CartGridField
import itertools


class OversetPart2D:
    def __init__(self, mpi: DNDS.MPIInfo, transform=(np.eye(3, 3), np.zeros(3))):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)
        self._transform = transform
        assert transform[0].shape == (3, 3)
        assert transform[1].shape == (3,)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if not (value[0].shape == (3, 3) and value[1].shape == (3,)):
            raise ValueError("value shape wrong " + f"{value}")
        rotrott = value[0] @ value[0].T
        err = np.linalg.norm(rotrott - np.eye(3), "fro")
        if err > 1e-5:
            raise ValueError(f"input not close to rotation: {value[0]}")
        if np.linalg.norm(rotrott * np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])) > 1e-5:
            raise ValueError(f"input not 2d rotation: {value[0]}")
        self._transform = value

        # don't forget to update AABB!
        coordsFatherData = np.array(self._mesh.coords.father.data())
        coordsFatherData = coordsFatherData.reshape((3, -1), order="F")
        coordsFatherData = self.coord_mesh_to_phy(coordsFatherData)
        coordsMax = coordsFatherData.max(axis=1)
        coordsMin = coordsFatherData.min(axis=1)
        self._MPI.Allreduce(None, coordsMax, op=pyMPI.MAX)
        self._MPI.Allreduce(None, coordsMin, op=pyMPI.MIN)
        self._xyzMax = coordsMax
        self._xyzMin = coordsMin

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
        self._MPI.Allreduce(None, coordsMax, op=pyMPI.MAX)
        self._MPI.Allreduce(None, coordsMin, op=pyMPI.MIN)
        self._xyzMax = coordsMax
        self._xyzMin = coordsMin

    @property
    def xyzMinMax(self):
        return np.concatenate(
            (self._xyzMin.transpose(), self._xyzMax.transpose())
        ).reshape((2, 3), order="C")

    def obtain_dist_node(self, bc_names=["WALL"]):
        from OversetCartUtil import obtain_part_local_elem_dists

        self.dist_node = obtain_part_local_elem_dists(self, bc_names)

    def get_travelling_cell(self, iCell: int, in_phy: bool = True):
        """_summary_

        Args:
            iCell (int): is local iCell
            in_phy (bool, optional): in physical instead of mesh space. Defaults to True.

        Returns:
            tuple[ElemType, int, int, list[int], ndarray]: travelling cell
        """
        mesh = self._mesh
        part = self

        cell2node = np.array(mesh.cell2node[iCell], copy=False)
        coords = []
        for iNode in cell2node:
            coords.append(np.array(mesh.coords[iNode], copy=True))
        coords = np.array(coords).transpose()
        if in_phy:
            coords = part.coord_mesh_to_phy(coords)
        elemInfo = mesh.cellElemInfo[iCell, 0]
        assert elemInfo.getElemType() in {
            Geom.Elem.ElemType.Tri3,
            Geom.Elem.ElemType.Quad4,
        }  # coords is now polygon
        travelling_cell = pack_travelling_cell(
            cellType=elemInfo.getElemType(),
            cellZone=elemInfo.zone,
            iCell=mesh.cell2node.trans.LGhostMapping(-1, iCell),
            cell2nodeRow=[
                mesh.coords.trans.LGhostMapping(-1, iNode) for iNode in cell2node
            ],
            coords=coords,
        )
        return travelling_cell

    def get_holed_faces_midpt(self, cell_type_arr: DNDS.ArrayEigenVectorPair_1_1_D):
        """_summary_

        Args:
            cell_type_arr (DNDS.ArrayEigenVectorPair_1_1_D): _description_

        Returns:
            (array, array): iFaces (local), midpt coords
        """
        mesh = self._mesh
        faces = []
        coords_mid = []
        for iFace in range(mesh.NumFaceProc()):
            f2c = mesh.face2cell[iFace].tolist()
            type_L = cell_type_arr[f2c[0]].tolist()[0]
            type_R = type_L
            if f2c[1] != DNDS.UnInitIndex:
                type_R = cell_type_arr[f2c[1]].tolist()[0]
            if type_L != type_R:
                faces.append(iFace)
                faceAtr = mesh.faceElemInfo[iFace, 0]
                assert faceAtr.getElemType() == Geom.Elem.ElemType.Line2
                f2n = mesh.face2node[iFace].tolist()
                coord_mid = (
                    np.array(mesh.coords[f2n[0]]) + np.array(mesh.coords[f2n[1]])
                ) * 0.5
                coord_mid = self.coord_mesh_to_phy(coord_mid)
                coords_mid.append(coord_mid[:2])
        coords_mid = np.array(coords_mid).reshape(-1, 2).T

        return (np.array(faces, dtype=np.int64), coords_mid)

    def print_full_mesh_type(
        self,
        iPart,
        cell_type_arr: DNDS.ArrayEigenVectorPair_1_1_D,
        no_hole=False,
        ax=None,
        highlight_iCells=set(),
    ):
        mpi = self._mpi
        MPI = self._MPI

        highlight_iCells = set([int(v) for v in highlight_iCells])
        highlight_iCells = MPI.gather(list(highlight_iCells), root=0)
        if mpi.rank == 0:
            highlight_iCells = set(itertools.chain(*highlight_iCells))

        mesh = self._mesh
        travelling_cell_type_list = []
        for iCell in range(mesh.NumCell()):
            cell_type = cell_type_arr[iCell].tolist()[0]  # float
            travelling_cell_type_list.append(
                self.get_travelling_cell(iCell, in_phy=True) + (cell_type,)
            )

        all_cell_types = MPI.gather(travelling_cell_type_list, root=0)

        if mpi.rank != 0:
            return

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        self_ax = ax is None
        if ax is None:
            fig = plt.figure(figsize=(8, 6), dpi=320)
            ax = plt.gca()
        for cellType, _, iCell, _, coords, osType in itertools.chain(*all_cell_types):
            assert cellType in {Geom.Elem.ElemType.Quad4, Geom.Elem.ElemType.Tri3}
            if no_hole and not (osType == 0):
                continue
            # if int(iCell) in highlight_iCells:
            #     print(f"HIGHLIGHT {iPart},{iCell}")
            polygon = patches.Polygon(
                coords[:2, :].transpose(),
                closed=True,
                alpha=0.8 if int(iCell) in highlight_iCells else 0.2,
                edgecolor="k",
                facecolor=f"C{int(iPart)}",
                ls="-" if osType == 0 else "none",
                lw=0.3,
            )
            ax.add_patch(polygon)

        if self_ax:
            plt.axis("equal")
            plt.savefig(f"part_{iPart}.png")
            plt.close(fig)


class DistMap:

    def __init__(
        self,
        cell_bnds: dict[tuple[int], list[t_travelling_cell_pack]],
        dist_field: CartGridField,
        cell_cell_inds: dict[tuple[int], set[int]],
        cell_cells_on_bnd: dict[tuple[int], list[t_travelling_cell_pack]],
        cell_bnds_expanded: dict[tuple[int], list[t_travelling_cell_pack]],
    ):
        (
            self.cell_bnds,  # ijk (g) to travelling boundary elements on the current cell
            self.dist_field,  # scalar field representing interpolated distance
            self.cell_cell_inds,  # ijk (g) to global cells intersecting
            self.cell_cells_on_bnd,  #  with all the cells on the bg cell having on zero cell_bnds
            self.cell_bnds_expanded,  # cell_bnds but with fringe records
        ) = (
            cell_bnds,
            dist_field,
            cell_cell_inds,
            cell_cells_on_bnd,
            cell_bnds_expanded,
        )


class OversetBG2D:

    def __init__(self, mpi: DNDS.MPIInfo):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)

    def set_bg(self, parts: list[OversetPart2D], h: float):
        mpi = self._mpi
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
        self.grid_shape = tuple([v.size for v in xNodes])  # shape for global point grid
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

        # for index conversion in DNDSR
        nPointsPerRank = np.array(
            [np.array(self.proc_grid_shape(rank)).prod() for rank in range(mpi.size)],
            dtype=np.int64,
        )
        self.global_map_pointsStart = np.concat(([0], np.cumsum(nPointsPerRank)))
        nCellsPerRank = np.array(
            [
                np.array(self.proc_cell_grid_shape(rank)).prod()
                for rank in range(mpi.size)
            ],
            dtype=np.int64,
        )
        self.global_map_cellsStart = np.concat(([0], np.cumsum(nCellsPerRank)))
        self.rank_to_ax_rank_map_i = np.array(
            [self.rank_to_ax_rank(rank)[0] for rank in range(mpi.size)], dtype=np.int32
        )
        self.rank_to_ax_rank_map_j = np.array(
            [self.rank_to_ax_rank(rank)[1] for rank in range(mpi.size)], dtype=np.int32
        )

        ## do borders
        ijk_ranges_point = self.proc_grid_range_expanded(is_point=True)
        # print(f"{mpi.rank}, { self.proc_grid_range(is_point=True)}, {ijk_ranges_point}")
        ism_expanded, jsm_expanded = np.meshgrid(
            *[
                np.arange(
                    ijk_ranges_point[iax][0],
                    ijk_ranges_point[iax][1],
                    dtype=np.int64,
                )
                for iax in range(2)
            ],
            indexing="ij",
        )
        self.local_point_grid_shape_expanded = ism_expanded.shape
        grid_point_expanded_idxs_g = self.ijk_to_idx(
            np.array([ism_expanded, jsm_expanded]),
            is_point=True,
        ).flat
        in_local = (
            grid_point_expanded_idxs_g >= self.global_map_pointsStart[mpi.rank]
        ) & (grid_point_expanded_idxs_g < self.global_map_pointsStart[mpi.rank + 1])
        grid_point_expanded_idxs_g = grid_point_expanded_idxs_g[~in_local]
        self.grid_point_expanded_idxs_g = np.sort(
            np.array(grid_point_expanded_idxs_g, dtype=np.int64)
        )
        self.grid_point_expanded_idxs_g_ijks_local = self.idx_to_ijk(
            self.grid_point_expanded_idxs_g, is_point=True, ret_global=True
        ) - np.array(
            [[self.proc_grid_range_expanded(is_point=True)[iax][0]] for iax in range(2)]
        )  # base point  # becomes in expanded array instead of in non-expanded

    def rank_to_ax_rank(self, rank=None):
        if rank is None:
            rank = self._mpi.rank
        return (rank // self.nProcAx[1], rank % self.nProcAx[1])

    def proc_cell_grid_shape(self, rank=None):
        ax_rank = self.rank_to_ax_rank(rank)
        return tuple(
            [
                self.nStarts[i][ax_rank[i] + 1] - self.nStarts[i][ax_rank[i]]
                for i in range(2)
            ]
        )

    def proc_grid_range(self, rank=None, is_point=True):
        nStarts_ax = self.nStarts4point if is_point else self.nStarts
        ax_rank = self.rank_to_ax_rank(rank)
        return [
            (nStarts_ax[i][ax_rank[i]], nStarts_ax[i][ax_rank[i] + 1]) for i in range(2)
        ]

    def proc_grid_shape(self, rank=None, is_point=True):
        ax_rank = self.rank_to_ax_rank(rank)
        nStarts_ax = self.nStarts4point if is_point else self.nStarts
        return tuple(
            [
                nStarts_ax[i][ax_rank[i] + 1] - nStarts_ax[i][ax_rank[i]]
                for i in range(2)
            ]
        )

    def proc_grid_range_expanded(self, rank=None, is_point=True):
        ijk_ranges = self.proc_grid_range(rank=rank, is_point=is_point)
        for iax in range(2):
            ijk_ranges[iax] = (
                max(ijk_ranges[iax][0] - 1, 0),
                min(
                    ijk_ranges[iax][1] + (2 if is_point else 1),
                    self.grid_shape[iax]
                    - (0 if is_point else 1),  # mind this parentheses
                ),
            )
        return ijk_ranges

    def proc_grid_shape_expanded(self, rank=None, is_point=True):
        ijk_ranges = self.proc_grid_range_expanded(rank=rank, is_point=is_point)
        return tuple([ijk_ranges[iax][1] - ijk_ranges[iax][0] for iax in range(2)])

    def proc_grid_core_start_in_local_expanded(self, rank=None, is_point=True):
        index_range = self.proc_grid_range(rank, is_point)
        index_range_expanded = self.proc_grid_range_expanded(rank, is_point)
        starts = [
            int(index_range[iax][0] - index_range_expanded[iax][0]) for iax in range(2)
        ]
        return starts

    def proc_grid_core_range_in_local_expanded(self, rank=None, is_point=True):
        starts = self.proc_grid_core_start_in_local_expanded(rank, is_point)
        return [
            (starts[iax], starts[iax] + int(self.proc_grid_shape(rank, is_point)[iax]))
            for iax in range(2)
        ]

    def global_idx_to_rank(self, idx, is_point=False):
        ranks = (
            np.searchsorted(
                (
                    self.global_map_pointsStart
                    if is_point
                    else self.global_map_cellsStart
                ),
                idx,
                side="right",
            )
            - 1
        )
        if np.any(ranks < 0) or np.any(ranks >= self._mpi.size):
            raise ValueError(f"idx {idx} out of range")
        return ranks

    def global_ijk_to_rank(self, ijk, is_point=False):
        if len(ijk) == 0:
            return []
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
                raise ValueError(
                    f"ijk {ijk[iax]} {ijk} {self.nStarts4point[iax] if is_point else self.nStarts[iax]} {indexAx} out of range"
                )
            idxs.append(indexAx)
        # print(idxs)
        return self.procMap[tuple(idxs)]

    def ijk_to_idx(
        self, ijk, is_point=False, in_global=True, ret_global=True, ranks=None
    ):
        if in_global and ranks is None:
            ranks = self.global_ijk_to_rank(ijk, is_point)
        if ranks is None and not in_global:
            raise ValueError("if input is local, should provide ranks")
        start_offsets = (
            self.global_map_pointsStart[ranks]
            if is_point
            else self.global_map_cellsStart[ranks]
        )
        ax_rank_is = self.rank_to_ax_rank_map_i[ranks]
        ax_rank_js = self.rank_to_ax_rank_map_j[ranks]
        nStarts_ax = self.nStarts4point if is_point else self.nStarts
        nStarts_ax_is = nStarts_ax[0][ax_rank_is]
        nStarts_ax_js = nStarts_ax[1][ax_rank_js]
        nStarts_ax_jlen = nStarts_ax[1][ax_rank_js + 1] - nStarts_ax_js
        ## conversion
        local_ijks = (
            ijk - np.array([nStarts_ax_is, nStarts_ax_js]) if in_global else ijk
        )
        local_idxs = local_ijks[0] * nStarts_ax_jlen + local_ijks[1]
        global_idxs = start_offsets + local_idxs
        return global_idxs if ret_global else local_idxs

    def idx_to_ijk(
        self, idx, is_point=False, in_global=True, ret_global=True, ranks=None
    ):
        if in_global and ranks is None:
            ranks = self.global_idx_to_rank(idx, is_point)
        if ranks is None and not in_global:
            raise ValueError("if input is local, should provide ranks")
        start_offsets = (
            self.global_map_pointsStart[ranks]
            if is_point
            else self.global_map_cellsStart[ranks]
        )
        ax_rank_is = self.rank_to_ax_rank_map_i[ranks]
        ax_rank_js = self.rank_to_ax_rank_map_j[ranks]
        nStarts_ax = self.nStarts4point if is_point else self.nStarts
        nStarts_ax_is = nStarts_ax[0][ax_rank_is]
        nStarts_ax_js = nStarts_ax[1][ax_rank_js]
        nStarts_ax_jlen = nStarts_ax[1][ax_rank_js + 1] - nStarts_ax_js
        ## conversion
        local_idxs = idx - start_offsets if in_global else idx
        local_is = local_idxs // nStarts_ax_jlen
        local_js = local_idxs % nStarts_ax_jlen
        if ret_global:
            ijks = np.array([nStarts_ax_is + local_is, nStarts_ax_js + local_js])
        else:
            ijks = np.array([local_is, local_js])
        return ijks

    def proc_grid_ijkarray(
        self, rank=None, is_point=False, expanded=False, no_mesh=False
    ):
        grid_range_method = (
            self.proc_grid_range_expanded if expanded else self.proc_grid_range
        )
        ijkRanges = grid_range_method(rank=rank, is_point=is_point)
        iss = np.arange(ijkRanges[0][0], ijkRanges[0][1], dtype=np.int64)
        jss = np.arange(ijkRanges[1][0], ijkRanges[1][1], dtype=np.int64)
        if no_mesh:
            return iss, jss
        ism, jsm = np.meshgrid(iss, jss, indexing="ij")
        ijks = np.array([ism, jsm])
        return ijks

    def proc_grid_point_coords(self, expanded=False):
        return self.origins[
            :2, np.newaxis, np.newaxis
        ] + self.h * self.proc_grid_ijkarray(is_point=True, expanded=expanded)

    @staticmethod
    @property
    def cell_bnds_type():
        return dict[tuple[int], list[tuple[Geom.Elem.ElemType, int, np.ndarray]]]

    @staticmethod
    @property
    def proc_dist_map_type():
        return DistMap

    def obtain_dist_map(self, parts: list[OversetPart2D], bc_names=["WALL"]):
        MPI = self._MPI
        mpi = self._mpi
        from OversetCartUtil import (
            obtain_proc_local_bg_dists,
            obtain_proc_local_bg_cell_bnd_elems,
            expand_proc_local_bg_cell_bnd_elems,
            obtain_proc_local_bg_cell_cell_elem_inds,
            obtain_proc_local_bg_cell_elems_with_bnd,
            get_mesh_bnd_elems,
            dist_field_neg_hole_expansion,
        )
        from CartUtil import single_elem_get_box_intersection_2D

        proc_dist_maps = []

        for iPart, part in enumerate(parts):
            proc_bg_mesh_dist = obtain_proc_local_bg_dists(self, part)
            cell_bnds = obtain_proc_local_bg_cell_bnd_elems(self, part)
            cell_bnds_expanded = expand_proc_local_bg_cell_bnd_elems(self, cell_bnds)
            cell_cell_inds = obtain_proc_local_bg_cell_cell_elem_inds(self, part)

            dist_field = CartGridField(1, self.proc_grid_shape(), mpi)
            dist_field.set_main_data(proc_bg_mesh_dist[:, :, None])
            dist_field.set_ghost_info(
                self.proc_grid_shape_expanded(is_point=True),
                self.proc_grid_core_range_in_local_expanded(is_point=True),
                self.grid_point_expanded_idxs_g_ijks_local,
            )
            dist_field.set_ghost_global_pull(self.grid_point_expanded_idxs_g)
            dist_field.trans.startPersistentPull()
            dist_field.trans.waitPersistentPull()
            # print(
            #     f"{mpi.rank} - {self.grid_point_expanded_idxs_g_ijks_local}, {dist_field.ghost.Size()}"
            # )

            dist_field_neg_hole_expansion(self, part, bc_names, cell_bnds, dist_field)

            cell_cells_on_bnd = obtain_proc_local_bg_cell_elems_with_bnd(
                self, part, cell_bnds, cell_cell_inds
            )

            new_dist_map = DistMap(
                cell_bnds,
                dist_field,
                cell_cell_inds,
                cell_cells_on_bnd,
                cell_bnds_expanded,
            )

            proc_dist_maps.append(new_dist_map)

        return proc_dist_maps

    def query_dist_from_points(self, distMap: DistMap, points: np.ndarray):
        from OversetCartUtil import query_dist_from_points

        return query_dist_from_points(self, distMap, points)

    def query_template_cell_from_points(
        self, osPart: OversetPart2D, distMap: DistMap, points: np.ndarray
    ):
        from OversetCartUtil import query_template_cell_from_points

        return query_template_cell_from_points(self, osPart, distMap, points)

    def decide_cell_types(
        self, parts: list[OversetPart2D], proc_dist_maps: list[DistMap]
    ):
        from OversetCartUtil import decide_cell_types

        return decide_cell_types(self, parts, proc_dist_maps)

    def decide_point_templates(
        self,
        parts: list[OversetPart2D],
        proc_dist_maps: list[DistMap],
        points: np.ndarray,
    ):
        from OversetCartUtil import decide_point_templates

        return decide_point_templates(
            self, parts, proc_dist_maps, points
        )  # iPartTemp,iCellTemp

    def print_proc_dist_maps(self, proc_dist_maps: list[DistMap], cmin=-1, cmax=10):
        mpi = self._mpi
        for iPart, item in enumerate(proc_dist_maps):
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            cell_bnds, dist_field, cell_cell_inds = (
                item.cell_bnds,
                item.dist_field,
                item.cell_cell_inds,
            )
            cell_cells_on_bnd = item.cell_cells_on_bnd

            data = np.array(dist_field.get_expanded_array()[:, :, 0])
            data = np.minimum(data, cmax)
            data = np.maximum(data, cmin)
            [xv, yv] = self.proc_grid_point_coords(expanded=True)
            # Create the plot

            fig = plt.figure(figsize=(8, 6), dpi=320)
            print(f"{xv.shape}, {yv.shape}, {data.shape}")
            qmesh = plt.pcolormesh(
                xv,
                yv,
                data,
                cmap="viridis",
                shading="gouraud",
                edgecolors="black",
                linewidths=0.5,
            )  # 'viridis' is a common colormap
            lw = 0.5
            for x in xv[:, 0]:
                plt.plot([x, x], [yv[0, 0], yv[0, -1]], c="k", lw=lw)
            for y in yv[0, :]:
                plt.plot([xv[-1, 0], xv[0, 0]], [y, y], c="k", lw=lw)
            for type, bcid, _, b2n, coords in itertools.chain(*cell_bnds.values()):
                assert type == Geom.Elem.ElemType.Line2
                plt.plot(coords[0], coords[1], c="r", marker=".", lw=0.1, ms=0.1)
            for ijks in cell_bnds.keys():
                center = (np.array(ijks) + 0.5) * self.h + self.origins[:2]
                plt.plot(
                    center[0], center[1], marker="o", ms=5, mfc="none", mew=0.3, c="k"
                )

            for ijks in cell_cell_inds:
                center = (np.array(ijks) + 0.5) * self.h + self.origins[:2]
                number = len(cell_cell_inds[ijks])
                plt.text(
                    center[0], center[1], f"{number}", va="center", ha="center", size=4
                )

            for ijks, elems in cell_cells_on_bnd.items():
                for cellType, cellZone, iCell, cell2nodeRow, coords in elems:
                    assert cellType in {
                        Geom.Elem.ElemType.Quad4,
                        Geom.Elem.ElemType.Tri3,
                    }
                    for iN in range(coords.shape[1]):
                        # plt.plot(
                        #     coords[0], coords[1], c="g", marker=".", lw=0.1, ms=0.1
                        # )
                        polygon = patches.Polygon(
                            coords[:2, :].transpose(),
                            closed=True,
                            edgecolor="none",
                            facecolor=(0.5, 0.2, 0, 0.3),
                            lw=1,
                        )
                        plt.gca().add_patch(polygon)

            # Add a colorbar to show the mapping of values to colors
            plt.colorbar()
            plt.axis("equal")
            plt.savefig(f"part_{iPart}_rank_{mpi.rank}.png")
            plt.close(fig)


if __name__ == "__main__":
    mpiGlob = DNDS.MPIInfo()
    mpiGlob.setWorld()
    from OversetCartManager import OversetBG2DManager

    translates = [[0, 0, 0], [1.5, 0, 0]]

    transforms = [
        (np.eye(3, 3), np.array(translates[i])) for i in range(len(translates))
    ]
    # translates = [[0, 0, 0]]

    mesh_names = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "mesh",
            "CylinderNoFar.cgns",
        )
    ] * len(transforms)

    osMan = OversetBG2DManager(mpiGlob)

    osMan.read_meshes_and_init(mesh_names, transforms)

    osMan.process_overset(1.0 / 10)

    osMan.print_proc_dist_maps()

    osMan.print_full_mesh_type(together=True)

    osBG = osMan.osBG
    if mpiGlob.rank == 0:
        _ = osBG.global_ijk_to_rank(
            np.random.randint(0, min(osBG.grid_shape[0:2]) - 1, size=(2, 32))
        )

        # print(osBG.global_ijk_to_rank([v - 1 for v in osBG.grid_shape], is_point=True))

        try:
            print(
                osBG.global_ijk_to_rank(
                    [v - 1 for v in osBG.grid_shape], is_point=False
                )
            )
        except ValueError as e:
            print(e)
            print("caught")
        else:
            raise RuntimeError("not raising value error")
