from OversetCart import *
from CartUtil import *
import scipy.spatial
from ElemInterpolate import elem_get_interpolation_base


def get_mesh_bnd_elems(osPart: OversetPart2D, includeIDs=None):
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
            nodeList.append(np.array(mesh.coords[n], copy=True))
        nodeList = np.array(nodeList).transpose()
        bnd_elems.append((bElemInfo.getElemType(), bElemInfo.zone, nodeList))
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
    mpi = self._mpi
    MPI = self._MPI
    bnd_elems_local = get_mesh_bnd_elems(
        self, [self._name2ID[name] for name in bc_names]
    )
    bnd_elems_local_others = MPI.allgather(bnd_elems_local)

    bnd_geoms = []
    for bnd_part in bnd_elems_local_others:
        for e_tuple in bnd_part:
            assert e_tuple[0] == Geom.Elem.ElemType.Line2
            bnd_geoms.extend([e_tuple[2][:, 0], e_tuple[2][:, 1]])
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
                assert (
                    g_point[ax] < self.nStarts4point[ax][ax_ranks[ax] + 1]
                ), g_point[ax]
            l_point = (
                g_point[0] - self.nStarts4point[0][ax_ranks[0]],
                g_point[1] - self.nStarts4point[1][ax_ranks[1]],
            )
            proc_bg_mesh_dist[l_point] = v
    print(f"proc {mpi.rank}: min dist at grid point: {proc_bg_mesh_dist.min()}")
    return proc_bg_mesh_dist
