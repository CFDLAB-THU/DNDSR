import DNDS, Geom
import numpy as np


class CartGridField:
    def __init__(
        self, rs: int, local_shape: tuple[int, int], mpi: DNDS.MPIInfo, rs_dyn=None
    ):
        self._mpi = mpi
        self._rs = rs
        self._local_shape = local_shape
        self._local_array_shape = local_shape + (rs,)
        local_size = np.array(local_shape, dtype=np.int64).prod()
        arrPair = DNDS.ArrayEigenVectorPair(rs)
        arrPair.father = DNDS.ArrayEigenVector(rs, init_args=(mpi,))
        arrPair.son = DNDS.ArrayEigenVector(rs, init_args=(mpi,))
        arrPair.TransAttach()

        arrPair.father.Resize(local_size)
        self.pair = arrPair

    @property
    def main(self):
        return self.pair.father

    @property
    def ghost(self):
        return self.pair.son

    @property
    def trans(self):
        return self.pair.trans

    @property
    def mdata(self):
        return np.array(self.main.data(), copy=False).reshape(self._local_array_shape)

    def set_main_data(self, data):
        dist_field = self
        main_mem = np.array(dist_field.main.data(), copy=False)
        main_mem.flat = data.flat

        assert (np.array(dist_field.main.data()) == data.flat).all()

    def set_main_data_from_expanded(self, data):
        dist_field = self
        cr = self._core_range_in_expanded
        main_mem = np.array(dist_field.main.data(), copy=False)
        main_mem.flat = data[cr[0][0] : cr[0][1], cr[1][0] : cr[1][1]].flat

        assert (
            np.array(dist_field.main.data())
            == data[cr[0][0] : cr[0][1], cr[1][0] : cr[1][1]].flat
        ).all()

    def set_ghost_global_pull(self, pullIdx):
        self.pair.trans.createFatherGlobalMapping()
        self.pair.trans.createGhostMapping(pullIdx)
        self.pair.trans.createMPITypes()
        self.pair.trans.initPersistentPull()

    def set_ghost_info(
        self,
        expanded_shape,
        core_range_in_expanded,
        grid_point_expanded_idxs_g_ijks_local,
    ):
        self._expanded_shape = expanded_shape
        self._expanded_array_shape = expanded_shape + (self._rs,)
        self._core_range_in_expanded = core_range_in_expanded
        self._grid_point_expanded_idxs_g_ijks_local = (
            grid_point_expanded_idxs_g_ijks_local
        )

    def get_expanded_array(self):
        mpi = self._mpi
        data_expanded = np.ones(self._expanded_array_shape) * 1e301
        cr = self._core_range_in_expanded
        ghost_is, ghost_js = self._grid_point_expanded_idxs_g_ijks_local

        data_expanded[cr[0][0] : cr[0][1], cr[1][0] : cr[1][1]] = self.mdata
        data_expanded[ghost_is, ghost_js] = np.array(self.ghost.data()).reshape(
            (-1, self._rs)
        )
        return data_expanded


# class CartGridIndTable:
#     def __init__(self, local_shape: tuple[int, int], mpi: DNDS.MPIInfo, rs_dyn=None):
#         self._mpi = mpi
#         self._local_shape = local_shape
#         local_size = np.array(local_shape, dtype=np.int64).prod()

#         arrPair = DNDS.ArrayAdjacencyPair("I")
#         arrPair.father = DNDS.ArrayAdjacency("I", init_args=(mpi,))
#         arrPair.son = DNDS.ArrayAdjacency("I", init_args=(mpi,))
#         arrPair.TransAttach()

#         arrPair.father.Resize(local_size)
#         self.pair = arrPair

#     @property
#     def main(self):
#         return self.pair.father

#     @property
#     def ghost(self):
#         return self.pair.son

#     @property
#     def trans(self):
#         return self.pair.trans

