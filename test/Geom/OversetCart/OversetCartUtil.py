from OversetCart import *
from CartUtil import *
from CartGridField import *
from GeomUtils import *
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
        nodeList = []  #!2d: list of node coords represent a bnd element
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
    dists = np.ones((mesh.NumNodeProc()), dtype=np.float64) * 1e300
    coordsFatherData = np.array(self._mesh.coords.father.data())
    coordsFatherData = coordsFatherData.reshape((3, -1), order="F")

    distq, idx = tree.query(coordsFatherData.T)
    dists[0 : mesh.NumNode()] = distq

    coordsSonData = (
        np.array(self._mesh.coords.son.data())
        if self._mesh.coords.son.Size()
        else np.zeros(0)  # why does empty son returns a NULL buffer?
    )
    coordsSonData = coordsSonData.reshape((3, -1), order="F")

    distq, idx = tree.query(coordsSonData.T)
    dists[mesh.NumNode() :] = distq
    print(f"{mpi.rank}: maxDist: {dists.max()}")
    return dists


def obtain_part_local_dists(self: OversetBG2D, osPart: OversetPart2D):
    mpi = self._mpi
    local_point_dists = obtain_part_local_inner_grid_points_dist_dict(self, osPart)

    # print(f"points covered: {mpi.rank}, {len(local_point_dists)}")

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
    # print(f"proc {mpi.rank}: min dist at grid point: {proc_bg_mesh_dist.min()}")
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
    return dict(cell_bnds)


def expand_proc_local_bg_cell_bnd_elems(self: OversetBG2D, cell_bnds: dict):
    mpi = self._mpi
    MPI = self._MPI

    cell_bnds = dict(cell_bnds)

    bg_cell_range = self.proc_grid_range(is_point=False)
    proc_grid_ijks_expanded = self.proc_grid_ijkarray(expanded=True).reshape(2, -1)
    ijk_fringe = []
    for iIJK in range(proc_grid_ijks_expanded.shape[1]):
        ijk = [int(proc_grid_ijks_expanded[iax, iIJK]) for iax in range(2)]
        ijkInCore = [
            bg_cell_range[iax][0] <= ijk[iax] < bg_cell_range[iax][1]
            for iax in range(2)
        ]
        if not (ijkInCore[0] and ijkInCore[1]):
            ijk_fringe.append(tuple(ijk))
    ijk_fringe = np.array(ijk_fringe, dtype=np.int64).T
    ranks = self.global_ijk_to_rank(ijk_fringe)
    req_lists = [[] for _ in range(mpi.size)]
    for i, rank in enumerate(ranks):
        req_lists[rank].append(ijk_fringe[:, i])
    req_cur = MPI.alltoall(req_lists)
    send_lists_fringe_cell_bnd = [[] for rank in range(mpi.size)]
    for rank_source, ijks in enumerate(req_cur):
        for ijk in ijks:
            ijk_t = tuple(int(v) for v in ijk)
            if ijk_t in cell_bnds:
                send_lists_fringe_cell_bnd[rank_source].append(
                    (ijk_t, cell_bnds[ijk_t])
                )
    recv_lists_fringe_cell_bnd = MPI.alltoall(send_lists_fringe_cell_bnd)

    for ijk_t, bnds_list in itertools.chain(*recv_lists_fringe_cell_bnd):
        assert ijk_t not in cell_bnds
        cell_bnds[ijk_t] = bnds_list

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


def query_dist_from_points(
    self: OversetBG2D, distMap: DistMap, points: np.ndarray, rel_tol=1e-8
):
    mpi = self._mpi
    MPI = self._MPI
    assert points.shape[0] == 2
    points = np.reshape(points, (2, -1))

    ijks = points_get_grid_cells_2D(self.origins[:2], self.h, points)
    ranks = self.global_ijk_to_rank(ijks, is_point=False)
    # print(ranks)

    query_send = [set() for _ in range(mpi.size)]
    for iq, rank in enumerate(ranks):
        query_send[rank].add(tuple([int(v) for v in ijks[:, iq]]))
    query_recv = MPI.alltoall(query_send)

    data_send = [{} for _ in range(mpi.size)]

    proc_grid_point_dists = distMap.dist_field.get_expanded_array()

    for rank_to, queries in enumerate(query_recv):
        for ijk in queries:
            ijk_local_p = tuple(
                [
                    ijk[iax] - self.proc_grid_range_expanded(is_point=True)[iax][0]
                    for iax in range(2)
                ]
            )
            dists = [
                proc_grid_point_dists[
                    ijk_local_p[0] + disp[0], ijk_local_p[1] + disp[1]
                ]
                for disp in [(0, 0), (1, 0), (0, 1), (1, 1)]
            ]
            dists = np.array(dists)
            additional = None
            if ijk in distMap.cell_bnds:
                bnd_elems = distMap.cell_bnds[ijk]
                assert ijk in distMap.cell_cells_on_bnd
                bnd_cell_elems = distMap.cell_cells_on_bnd[ijk]
                additional = (bnd_elems, bnd_cell_elems)

                # only boundary-cut bg-cells check neighbour bg-cell for bnd elem
                ijk_neighs = [
                    (ijk[0] - 1, ijk[1]),
                    (ijk[0] + 1, ijk[1]),
                    (ijk[0], ijk[1] - 1),
                    (ijk[0], ijk[1] + 1),
                ]
                for ijk_neigh in ijk_neighs:
                    if (
                        ijk_neigh in distMap.cell_bnds_expanded
                    ):  # no need to ensure neigh is a valid ijk
                        additional[0].extend(distMap.cell_bnds_expanded[ijk_neigh])

            assert ijk not in data_send[rank_to]
            data_send[rank_to][ijk] = (dists, additional)

    data_recv = MPI.alltoall(data_send)
    data_recv_merged = {k: v for d in data_recv for k, v in d.items()}

    dist_results = np.zeros(points.shape[1]) + 1e300
    for iq in range(points.shape[1]):
        ijkt = tuple([int(v) for v in ijks[:, iq]])
        dists, additional = data_recv_merged[ijkt]
        if dists.max() < 0:
            dist_results[iq] = dists.max()
            continue
        if dists.min() > 1e299:
            dist_results[iq] = 1e300
            continue
        if ((0 <= dists) & (dists <= 1e299)).all():
            xyz01 = [
                (
                    self.origins[iax] + ijkt[iax] * self.h,
                    self.origins[iax] + (ijkt[iax] + 1) * self.h,
                )
                for iax in range(2)
            ]
            xis = np.array(
                [(points[iax, iq] - xyz01[iax][0]) / self.h for iax in range(2)],
                dtype=np.float64,
            )
            assert (
                (0 - rel_tol <= xis) & (xis <= 1 + rel_tol)
            ).all(), f"{xis}, {points[:, iq]}"

            bases = [
                np.array(
                    [1 - xis[iax] if offset[iax] == 0 else xis[iax] for iax in range(2)]
                ).prod()
                for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]
            ]
            dist_results[iq] = np.array(bases).dot(dists)
            continue
        assert additional is not None, f"{ijkt}, {iq}, {(dists.min(),dists.max())}"
        bnd_elems, bnd_cell_elems = additional
        found_num = 0
        for ct, _, _, _, coords in bnd_cell_elems:  #!check if inside mesh really
            is_in = points_in_polygon_winding(
                coords[:2, :], points[0, iq], points[1, iq]
            )
            if is_in:
                found_num += 1
                break
        if found_num:
            if dists.min() < 0:
                #! exact search!
                # todo: better exact search
                distc = 1e300
                for bct, _, _, _, coords in bnd_elems:
                    p = points[:, iq]
                    edist = np.linalg.vector_norm(
                        coords[:2, :] - p[:, None], axis=0
                    ).min()
                    distc = min(distc, edist)
                    p1 = coords[:2, 0]
                    p2 = coords[:2, 1]
                    x12 = p2 - p1
                    x10 = p - p1
                    xi = x10.dot(x12) / x12.dot(x12)
                    if 0 <= xi and xi <= 1:
                        distM = np.linalg.vector_norm(xi * x12 - x10)
                        distc = min(distc, distM)

                dist_results[iq] = distc
                # print(distc)
            if dists.max() > 1e299:
                dist_results[iq] = dists.min()
        else:
            if dists.min() < 0:
                dist_results[iq] = dists.min()
            if dists.max() > 1e299:
                dist_results[iq] = 1e300

        # raise RuntimeWarning("not implemented")

    # print(data_recv_merged)
    return dist_results


def query_template_cell_from_points(
    self: OversetBG2D, osPart: OversetPart2D, distMap: DistMap, points: np.ndarray
):
    mpi = self._mpi
    MPI = self._MPI
    mesh = osPart._mesh
    points = np.reshape(points, (2, -1))

    ijks = points_get_grid_cells_2D(self.origins[:2], self.h, points)
    ranks = self.global_ijk_to_rank(ijks, is_point=False)

    query_send = [set() for _ in range(mpi.size)]
    for iq, rank in enumerate(ranks):
        query_send[rank].add(tuple([int(v) for v in ijks[:, iq]]))
    query_recv = MPI.alltoall(query_send)

    data_send = [{} for _ in range(mpi.size)]
    for rank, query_recv_rank in enumerate(query_recv):
        for ijk in query_recv_rank:
            data_send[rank][ijk] = distMap.cell_cell_inds[ijk]
    data_recv = MPI.alltoall(data_send)
    ijk_to_cell_data_recv_merged = {k: v for d in data_recv for k, v in d.items()}

    iCellGs = set()
    for v in ijk_to_cell_data_recv_merged.values():
        iCellGs.update(v)
    cell_req_send = [[] for _ in range(mpi.size)]
    for iCellG in iCellGs:
        ret, rank, iCell = mesh.cell2node.trans.LGlobalMapping.search(iCellG)
        assert ret, "search failed"
        # print(rank)
        cell_req_send[rank].append((iCell, iCellG))
    cell_req_recv = MPI.alltoall(cell_req_send)
    travelling_cell_data_send = []
    for cell_reqs in cell_req_recv:
        data_c = []
        for iCell, iCellG in cell_reqs:
            data_c.append((iCellG, osPart.get_travelling_cell(iCell, in_phy=True)))
        travelling_cell_data_send.append(data_c)
    travelling_cell_data_recv = MPI.alltoall(travelling_cell_data_send)
    travelling_cell_data_recv_dict = {}
    for iCellG, travelling_cell in itertools.chain(*travelling_cell_data_recv):
        travelling_cell_data_recv_dict[iCellG] = travelling_cell

    template_iCellGs = []
    for iq, rank in enumerate(ranks):
        ijk_t = tuple([int(v) for v in ijks[:, iq]])
        cells = [
            (iCellG, travelling_cell_data_recv_dict[iCellG])
            for iCellG in ijk_to_cell_data_recv_merged[ijk_t]
        ]
        iCellG_template = -1
        for iCellG, travelling_cell in cells:
            cellType, cellZone, iCell, cell2nodeRow, coords = travelling_cell
            assert cellType in {
                Geom.Elem.ElemType.Quad4,
                Geom.Elem.ElemType.Tri3,
            }
            is_in = points_in_polygon_winding(
                coords[:2, :], points[0, iq], points[1, iq]
            )
            if is_in:
                iCellG_template = iCellG
        # assert iCellG_template >= 0, "template query not found"
        template_iCellGs.append(iCellG_template)
    return template_iCellGs


def decide_cell_types(
    self: OversetBG2D, parts: list[OversetPart2D], proc_dist_maps: list[DistMap]
):
    mpi = self._mpi
    MPI = self._MPI
    check_pairs = []
    for i in range(len(parts)):
        for j in range(len(parts)):
            check_pairs.append((i, j))

    other_dists_nodes = [{} for _ in range(len(parts))]
    self_dist_nodes = [None] * len(parts)

    for i, j in check_pairs:
        part_this = parts[i]
        dist_that = proc_dist_maps[j]
        mesh = part_this._mesh
        coordsFatherData = np.array(mesh.coords.father.data())
        coordsFatherData = coordsFatherData.reshape((3, -1), order="F")

        coordsSonData = (
            np.array(mesh.coords.son.data())
            if mesh.coords.son.Size()
            else np.zeros(0)  # why does empty son returns a NULL buffer?
        )
        coordsSonData = coordsSonData.reshape((3, -1), order="F")

        coordsFullData = np.concat([coordsFatherData, coordsSonData], axis=1)
        coordsFullData = part_this.coord_mesh_to_phy(
            coordsFullData
        )  # do not forget to physical coord
        coordsFullData = coordsFullData[:2, :]

        if j == i:
            self_dist_nodes[i] = self.query_dist_from_points(dist_that, coordsFullData)
        else:
            other_dists_nodes[i][j] = self.query_dist_from_points(
                dist_that, coordsFullData
            )

    min_dists_nodes_other = []
    for i, other_set in enumerate(other_dists_nodes):
        part_this = parts[i]
        mesh = part_this._mesh
        min_dist_other = np.zeros((mesh.NumNodeProc())) + 1e301
        for other_dist in other_set.values():
            min_dist_other = np.minimum(min_dist_other, other_dist)
            # print("===\n" * 3 + f"i{i}, {other_dist.min()}")
        min_dists_nodes_other.append(min_dist_other)

    cell_type_arrs = []

    for i, part in enumerate(parts):
        node_is_hole = (
            self_dist_nodes[i] * 0.9
            > min_dists_nodes_other[i]
            # part.dist_node > min_dists_nodes_other[i]
        )
        # !using self_dist_nodes[i], not using exact nodal values here, for field consistency

        # print("===\n" * 3 + f"i{i}, {node_is_hole.sum()}, {min_dists_nodes_other[i].min()}")
        mesh = part._mesh
        cell_type = np.zeros((mesh.NumCell()))
        for iCell in range(mesh.NumCell()):
            c2n = mesh.cell2node[iCell].tolist()
            c2n_is_hole = node_is_hole[c2n]
            cell_type[iCell] = 1 if c2n_is_hole.all() else 0
        cell_type_arr = DNDS.ArrayEigenVectorPair(1)
        cell_type_arr.father = DNDS.ArrayEigenVector(1, init_args=(mpi,))
        cell_type_arr.son = DNDS.ArrayEigenVector(1, init_args=(mpi,))
        cell_type_arr.TransAttach()
        cell_type_arr.father.Resize(mesh.NumCell())
        cell_type_arr_father_data_in = np.array(cell_type_arr.father.data(), copy=False)
        cell_type_arr_father_data_in[:] = cell_type.flat
        cell_type_arr.trans.BorrowGGIndexing(mesh.cell2node.trans)
        cell_type_arr.trans.createMPITypes()
        cell_type_arr.trans.pullOnce()
        cell_type_arrs.append(cell_type_arr)
    return (cell_type_arrs, other_dists_nodes)


def decide_point_templates(
    self: OversetBG2D,
    parts: list[OversetPart2D],
    proc_dist_maps: list[DistMap],
    points: np.ndarray,
):
    mpi = self._mpi
    MPI = self._MPI
    points = np.reshape(points, (2, -1))

    assert len(parts) == len(proc_dist_maps)

    dists_pts = []
    for dist_that in proc_dist_maps:
        dists_pts.append(self.query_dist_from_points(dist_that, points))
    # print(dists_pts)
    dist_pts_a = np.array(dists_pts, dtype=np.float64)
    min_dist_loc = np.argmin(dist_pts_a, axis=0)
    min_dist = np.min(dist_pts_a, axis=0)
    inds = np.arange(len(min_dist))
    iCellGs = np.ones_like(inds) * -1  # at iPart == min_dist_loc

    for i, part in enumerate(parts):
        inds_c = inds[min_dist_loc == i]
        points_c = points[:, inds_c]
        iCellGs[inds_c] = self.query_template_cell_from_points(
            part, proc_dist_maps[i], points_c
        )

    return (np.array(min_dist_loc), np.array(iCellGs))
