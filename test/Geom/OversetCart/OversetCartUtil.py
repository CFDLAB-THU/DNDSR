from OversetCart import *
from CartUtil import *
from CartGridField import *
from utils import *
import scipy.spatial
from ElemInterpolate import elem_get_interpolation_base
from collections import defaultdict


def get_mesh_bnd_elems(osPart: OversetPart2D, includeIDs=None, is_phy=False):
    mesh = osPart._mesh
    bnd_elems = []
    for iB in range(mesh.NumBnd()):
        bnd2node = np.array(osPart._mesh.bnd2node[iB], copy=False)
        bElemInfo = mesh.bndElemInfo[iB, 0]
        if includeIDs is not None and bElemInfo.zone not in includeIDs:
            continue
        assert (bnd2node < mesh.NumNodeProc()).all(), str(bnd2node)
        nodeList = []  #!2d: list of nodes represent a bnd element
        for n in bnd2node:
            if is_phy:
                nodeList.append(
                    osPart.coord_mesh_to_phy(np.array(mesh.coords[n], copy=True))
                )
            else:
                nodeList.append(np.array(mesh.coords[n], copy=True))
        nodeList = np.array(nodeList).transpose()
        travelling_bnd_elem = pack_travelling_cell(
            cellType=bElemInfo.getElemType(),
            cellZone=bElemInfo.zone,
            iCell=-1,  # set as unknown
            cell2nodeRow=[mesh.coords.trans.LGhostMapping(-1, n) for n in bnd2node],
            coords=nodeList,
        )
        # bnd_elems.append((bElemInfo.getElemType(), bElemInfo.zone, nodeList))
        bnd_elems.append(travelling_bnd_elem)
    return bnd_elems


def obtain_part_local_inner_grid_points_dist_dict(
    self: OversetBG2D, osPart: OversetPart2D
):
    local_point_dists = {}
    mesh = osPart._mesh
    origin = np.array(self.origins[0:2])
    h = self.h
    for iCell in range(mesh.NumCell()):
        cell2node = np.array(mesh.cell2node[iCell], copy=False)
        assert (cell2node < mesh.NumNodeProc()).all(), str(cell2node)
        nodeDists = osPart.dist_node[cell2node]
        coords = []
        for iNode in cell2node:
            coords.append(np.array(mesh.coords[iNode], copy=True))
        coords = np.array(coords).transpose()
        coords = osPart.coord_mesh_to_phy(coords)
        elemInfo = mesh.cellElemInfo[iCell, 0]
        assert elemInfo.getElemType() in {
            Geom.Elem.ElemType.Tri3,
            Geom.Elem.ElemType.Quad4,
        }  # coords is now polygon
        gridPoints = single_elem_get_grid_point_2D(origin, h, coords[0:2, :])
        gridPointsCoords = origin[:, None] + gridPoints * h

        if gridPoints.size:
            for iP in range(gridPoints.shape[1]):

                p = np.zeros(3)
                p[:2] = gridPointsCoords[:, iP]
                distP = elem_get_interpolation_base(
                    elemInfo.getElemType(), coords, p
                ).dot(nodeDists)
                local_point_dists[tuple(gridPoints[:, iP])] = distP
    return local_point_dists


def obtain_part_local_elem_dists(self: OversetPart2D, bc_names=["WALL"]):
    """for a part to get nodal distance field

    Args:
        self (OversetPart2D): _description_
        bc_names (list, optional): _description_. Defaults to ["WALL"].

    Returns:
        _type_: _description_
    """
    mpi = self._mpi
    MPI = self._MPI
    bnd_elems_local = get_mesh_bnd_elems(
        self, [self._name2ID[name] for name in bc_names]
    )
    bnd_elems_local_others = MPI.allgather(bnd_elems_local)

    bnd_geoms = []
    for bnd_part in bnd_elems_local_others:
        for t, z, _, b2n, coord in bnd_part:
            assert t == Geom.Elem.ElemType.Line2
            bnd_geoms.extend([coord[:, 0], coord[:, 1]])
    tree = scipy.spatial.KDTree(np.array(bnd_geoms))

    mesh = self._mesh
    dists = np.ones((mesh.NumNodeProc()), dtype=np.float64) * 1e-300
    coordsFatherData = np.array(self._mesh.coords.father.data())
    coordsFatherData = coordsFatherData.reshape((3, -1), order="F")

    distq, idx = tree.query(coordsFatherData.T)
    dists[0 : mesh.NumNode()] = distq

    coordsSonData = np.array(self._mesh.coords.son.data())
    coordsSonData = coordsSonData.reshape((3, -1), order="F")

    distq, idx = tree.query(coordsSonData.T)
    dists[mesh.NumNode() :] = distq
    print(f"{mpi.rank}: maxDist: {dists.max()}")
    return dists


def obtain_part_local_dists(self: OversetBG2D, osPart: OversetPart2D):
    mpi = self._mpi
    MPI = self._MPI
    origin = np.array(self.origins[0:2])
    h = self.h
    mesh = osPart._mesh
    local_point_dists = obtain_part_local_inner_grid_points_dist_dict(self, osPart)

    print(f"points covered: {mpi.rank}, {len(local_point_dists)}")

    return local_point_dists


def obtain_proc_local_bg_dists(self: OversetBG2D, part: OversetPart2D):
    mpi = self._mpi
    MPI = self._MPI
    part_local_dists = obtain_part_local_dists(self, part)
    part_local_dists_items = list(part_local_dists.items())

    ijkv = np.array([v[0] for v in part_local_dists_items]).transpose()
    if ijkv.size:
        ranks = list(self.global_ijk_to_rank(ijkv))
    else:
        ranks = []
    # sendLists = [[]] * self._mpi.size # ! this is wrong!!! the empty lists of each item refs the same
    sendLists = [[] for _ in range(mpi.size)]

    for i, rank in enumerate(ranks):
        datac = part_local_dists_items[i]
        sendLists[rank].append(datac)

    recvLists = MPI.alltoall(sendLists)
    # print(f"{mpi.rank}, {([len(v) for v in sendLists])}, {len(sendLists)}")
    # return

    ax_ranks = self.rank_to_ax_rank()
    proc_bg_mesh_dist = np.ones(self.proc_grid_shape()) * 1e300
    for recvL in recvLists:
        # print(recvL)
        for g_point, v in recvL:
            for ax in range(2):
                assert g_point[ax] < self.nStarts4point[ax][ax_ranks[ax] + 1], g_point[
                    ax
                ]
            l_point = (
                g_point[0] - self.nStarts4point[0][ax_ranks[0]],
                g_point[1] - self.nStarts4point[1][ax_ranks[1]],
            )
            proc_bg_mesh_dist[l_point] = v
    print(f"proc {mpi.rank}: min dist at grid point: {proc_bg_mesh_dist.min()}")
    return proc_bg_mesh_dist


def dist_field_neg_hole_expansion(
    self: OversetBG2D,
    part: OversetPart2D,
    bc_names: list[str],
    cell_bnds: OversetBG2D.cell_bnds_type,
    dist_field: CartGridField,
):
    """creates negative hole and expands it

    Args:
        self (OversetBG2D): _description_
        part (OversetPart2D): _description_
        bc_names (list[str]): _description_
        cell_bnds (OversetBG2D.cell_bnds_type): _description_
        dist_field (CartGridField): modified to have neg hole
    """
    mpi = self._mpi
    MPI = self._MPI

    dist_expanded_array = dist_field.get_expanded_array()

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for ijk, bnd_list in cell_bnds.items():
        for type, bcid, _, b2n, coords in bnd_list:
            assert type == Geom.Elem.ElemType.Line2
            if bcid in [part._name2ID[name] for name in bc_names]:
                for offset in offsets:
                    ijkp = tuple([ijk[iax] + offset[iax] for iax in range(2)])
                    ijkp_in_expanded = tuple(
                        [
                            ijkp[iax]
                            - self.proc_grid_range_expanded(is_point=True)[iax][0]
                            for iax in range(2)
                        ]
                    )
                    if dist_expanded_array[ijkp_in_expanded] > 1e299:
                        dist_expanded_array[ijkp_in_expanded] = -100
    dist_field.set_main_data_from_expanded(dist_expanded_array)
    dist_field.trans.startPersistentPull()
    dist_field.trans.waitPersistentPull()

    ijks_expanded = self.proc_grid_ijkarray(is_point=True, expanded=True)
    ijks_expanded[0] -= self.proc_grid_range_expanded()[0][0]
    ijks_expanded[1] -= self.proc_grid_range_expanded()[1][0]
    cr = self.proc_grid_core_range_in_local_expanded(is_point=True)
    ijks_expanded_core = ijks_expanded[:, cr[0][0] : cr[0][1], cr[1][0] : cr[1][1]]
    ijks_expanded_core = ijks_expanded_core.reshape((2, -1))
    ijks_expanded_core_le = np.array([ijks_expanded_core[0] - 1, ijks_expanded_core[1]])
    ijks_expanded_core_ri = np.array([ijks_expanded_core[0] + 1, ijks_expanded_core[1]])
    ijks_expanded_core_lo = np.array([ijks_expanded_core[0], ijks_expanded_core[1] - 1])
    ijks_expanded_core_up = np.array([ijks_expanded_core[0], ijks_expanded_core[1] + 1])

    offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for iter in range(10000):
        dist_expanded_array = dist_field.get_expanded_array()
        dist_expanded_array_new = np.array(dist_expanded_array)
        count = 0
        for ic in range(ijks_expanded_core.shape[1]):
            ijk = tuple(ijks_expanded_core[:, ic])
            if dist_expanded_array[ijk] > 1e299:
                for ijk_nei in [
                    ijks_expanded_core_le[:, ic],
                    ijks_expanded_core_ri[:, ic],
                    ijks_expanded_core_lo[:, ic],
                    ijks_expanded_core_up[:, ic],
                ]:
                    ijk_nei = tuple(ijk_nei)
                    if (
                        0 <= ijk_nei[0] < dist_expanded_array.shape[0]
                        and 0 <= ijk_nei[1] < dist_expanded_array.shape[1]
                        and dist_expanded_array[ijk_nei] < 0
                    ):
                        count += 1
                        dist_expanded_array_new[ijk] = dist_expanded_array[ijk_nei]
        count = MPI.allreduce(count, pyMPI.SUM)
        if mpi.rank == 0:
            print(f"negative hole: iter [{iter}] Done")
        if count == 0:
            break
        dist_field.set_main_data_from_expanded(dist_expanded_array_new)
        dist_field.trans.startPersistentPull()
        dist_field.trans.waitPersistentPull()
    else:
        print(f"count left: {count}")


def obtain_proc_local_bg_cell_elems_with_bnd(
    self: OversetBG2D,
    part: OversetPart2D,
    cell_bnds: OversetBG2D.cell_bnds_type,
    cell_cell_inds: dict[tuple[int], set[int]],
):
    mpi = self._mpi
    MPI = self._MPI
    mesh = part._mesh

    elem_reqs_send = [[] for _ in range(mpi.size)]
    for ijk in cell_bnds.keys():
        assert ijk in cell_cell_inds.keys()  # have at least 1 elem adjacent to bnd
        for iCellG in cell_cell_inds[ijk]:
            ret, rank, val = mesh.cell2node.trans.LGlobalMapping.search(iCellG)
            assert ret, "search failed"
            elem_reqs_send[rank].append((ijk, iCellG))

    elem_reqs_recv = MPI.alltoall(elem_reqs_send)

    elem_send = [[] for _ in range(mpi.size)]
    for rank_send, reqs in enumerate(elem_reqs_recv):
        for ijk, iCellG in reqs:
            ret, rank_searched, iCell = mesh.cell2node.trans.LGlobalMapping.search(
                iCellG
            )
            assert ret and rank_searched == mpi.rank, "search failed"
            elem_send[rank_send].append(
                (ijk, part.get_travelling_cell(iCell, in_phy=True))
            )
    elem_recv = MPI.alltoall(elem_send)
    # if mpi.rank == 0:
    #     print(elem_recv)
    cell_cells_on_bnd = defaultdict(list)
    for ijk, travelling_cell in itertools.chain(*elem_recv):
        cell_cells_on_bnd[ijk].append(travelling_cell)
    return cell_cells_on_bnd


def obtain_proc_local_bg_cell_bnd_elems(self: OversetBG2D, part: OversetPart2D):
    mpi = self._mpi
    MPI = self._MPI

    bnd_elems = get_mesh_bnd_elems(part, is_phy=True)  # note we are using phy
    send_lists = [[] for _ in range(mpi.size)]
    for travelling_bnd_elem in bnd_elems:
        type, bcid, _, b2n, coords = travelling_bnd_elem
        cell_ijks = single_elem_get_box_intersection_2D(
            self.origins[:2], self.h, coords[0:2, :]
        )
        ranks = self.global_ijk_to_rank(cell_ijks)
        for ic, rank in enumerate(ranks):
            send_lists[rank].append((cell_ijks[:, ic], travelling_bnd_elem))
    recv_lists = MPI.alltoall(send_lists)

    cell_bnds = defaultdict(list)
    for cell_ijk, travelling_bnd_elem in itertools.chain(*recv_lists):
        cell_ijk_t = tuple(int(v) for v in cell_ijk)
        cell_bnds[cell_ijk_t].append(travelling_bnd_elem)
    return cell_bnds


def obtain_proc_local_bg_cell_cell_elem_inds(self: OversetBG2D, part: OversetPart2D):

    mpi = self._mpi
    MPI = self._MPI
    mesh = part._mesh
    origin = self.origins[:2]

    send_list = [[] for _ in range(mpi.size)]

    for iCell in range(mesh.NumCell()):
        cell2node = np.array(mesh.cell2node[iCell], copy=False)
        assert (cell2node < mesh.NumNodeProc()).all(), str(cell2node)
        coords = []
        for iNode in cell2node:
            coords.append(np.array(mesh.coords[iNode], copy=True))
        coords = np.array(coords).transpose()
        coords = part.coord_mesh_to_phy(coords)
        elemInfo = mesh.cellElemInfo[iCell, 0]
        assert elemInfo.getElemType() in {
            Geom.Elem.ElemType.Tri3,
            Geom.Elem.ElemType.Quad4,
        }  # coords is now polygon
        # travelling_cell = pack_travelling_cell(
        #     cellType=elemInfo.getElemType(),
        #     cellZone=elemInfo.zone,
        #     iCell=mesh.cell2node.trans.LGhostMapping(-1, iCell),
        #     cell2nodeRow=[
        #         mesh.coords.trans.LGhostMapping(-1, iNode) for iNode in cell2node
        #     ],
        #     coords=coords,
        # )
        # ! we only need indices here not whole travelling cell
        ijks = single_elem_get_box_intersection_2D(origin, self.h, coords[:2, :])
        ranks = self.global_ijk_to_rank(ijks, is_point=False)
        for iC in range(ijks.shape[1]):
            send_list[ranks[iC]].append(
                (ijks[:, iC], mesh.cell2node.trans.LGhostMapping(-1, iCell))
            )

    recv_list = MPI.alltoall(send_list)
    proc_cell_global_ijk_to_part_cell_dict = defaultdict(list)
    d = proc_cell_global_ijk_to_part_cell_dict
    for ijk, iCellG in itertools.chain(*recv_list):
        ijkt = tuple([int(v) for v in ijk])
        d[ijkt].append(iCellG)
        
    for k in d:
        d[k] = set(sorted(d[k]))

    # if mpi.rank == 0:
    #     print(d)

    return proc_cell_global_ijk_to_part_cell_dict
