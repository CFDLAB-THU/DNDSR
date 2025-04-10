import DNDS, Geom
import mpi4py as pyMPI
from OversetCart import OversetBG2D, OversetPart2D, DistMap
from utils import get_mpi4py_comm_from_MPIInfo
import numpy as np
import sys, os


class OversetBG2DManager:
    def __init__(self, mpi: DNDS.MPIInfo):
        self._mpi = mpi
        self._MPI = get_mpi4py_comm_from_MPIInfo(mpi)

    def read_meshes_and_init(
        self,
        mesh_names: list[str],
        initial_transforms=list[tuple[np.ndarray, np.ndarray]],
    ):
        mpi = self._mpi
        assert len(mesh_names) == len(initial_transforms)
        self.parts = parts = [
            OversetPart2D(mpi, transform=initial_transforms[i])
            for i in range(len(initial_transforms))
        ]
        for i, osPart in enumerate(parts):
            osPart.read_mesh(mesh_names[i])
            osPart.obtain_dist_node()
        self.osBG = OversetBG2D(mpi)

    @property
    def nPart(self):
        return len(self.parts)

    def process_overset(self, h: float):
        mpi = self._mpi
        parts = self.parts
        osBG = self.osBG
        osBG.set_bg(parts, h)
        assert osBG.procMap[osBG.rank_to_ax_rank()] == mpi.rank
        self.dist_maps = dist_maps = osBG.obtain_dist_map(parts)
        # self.print_proc_dist_maps()
        self.cell_type_arrs, self.other_dists_nodes = osBG.decide_cell_types(
            parts, dist_maps
        )

        print(
            osBG.query_dist_from_points(
                dist_maps[0], points=np.array([[0.25, 0, 0 - 0.025], [0.5, 0.25, 0.5]])
            )
        )

    def print_proc_dist_maps(self):
        self.osBG.print_proc_dist_maps(self.dist_maps)

    def print_full_mesh_type(self, out_name="", together=False):
        if not together:
            for iPart, part in enumerate(self.parts):
                part.print_full_mesh_type(iPart, self.cell_type_arrs[iPart])
        else:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 6), dpi=400)
            for iPart, part in enumerate(self.parts):
                part.print_full_mesh_type(
                    iPart, self.cell_type_arrs[iPart], no_hole=True, ax=plt.gca()
                )
            plt.axis("equal")
            plt.gca().autoscale()
            if len(out_name):
                out_name += "_"
            plt.savefig(out_name + "parts_all.png")
            plt.close(fig)


if __name__ == "__main__":
    mpiGlob = DNDS.MPIInfo()
    mpiGlob.setWorld()

    translates = [[0, 0, 0], [1.5, 0, 0]]
    translates_end = [[0, 0, 0], [0.5, 0, 0]]
    translates = np.array(translates)
    translates_end = np.array(translates_end)
    Nstep = 41

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

    # for iter in range(100000):
    #     osMan.process_overset(1.0 / 10) #todo: test forged leak here

    inter = np.linspace(0, 1, Nstep)
    for i in range(0, Nstep):
        translate_now = inter[i] * translates_end + (1 - inter[i]) * translates
        transforms = [
            (np.eye(3, 3), np.array(translate_now[i]))
            for i in range(len(translate_now))
        ]
        for iPart, part in enumerate(osMan.parts):
            part.transform = transforms[iPart]

        osMan.process_overset(1.0 / 10)

        osMan.print_full_mesh_type(together=True, out_name=f"os_type_{i:04}")
