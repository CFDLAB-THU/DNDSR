# import DNDS
# import Geom


from DNDSR import DNDS as DNDS
from DNDSR import Geom as Geom
from DNDSR.Geom.utils import create_mesh_from_CGNS, create_bnd_mesh
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

print(DNDS.__file__)


def test_mesh0():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    mesh, reader, name2Id = create_mesh_from_CGNS(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
        ),
        mpi,
        2
    )

    meshBnd, readerBnd = create_bnd_mesh(mesh)

    # fig, ax = plt.subplots(figsize=(16, 16), dpi=320)
    # xymaxs = np.array([-1e100, -1e100], dtype=np.double)
    # xymins = np.array([1e100, 1e100], dtype=np.double)
    # for iCell in range(mesh.cell2node.Size()):
    #     c2n = mesh.cell2node[iCell].tolist()
    #     nodes = []
    #     # print(mesh.cell2node[iCell].tolist())

    #     for iNode in c2n:
    #         nodes.append(np.array(mesh.coords[iNode]))
    #     nodes = np.array(nodes)

    #     vertices = nodes[:, 0:2]
    #     xymaxs = np.maximum(vertices.max(axis=0), xymaxs)
    #     xymins = np.minimum(vertices.min(axis=0), xymins)
    #     # print(vertices.max(axis=0))
    #     polygon = patches.Polygon(
    #         vertices,
    #         closed=True,
    #         edgecolor="black",
    #         facecolor="blue" if iCell < mesh.cell2node.father.Size() else "red",
    #         lw=1,
    #     )
    #     ax.add_patch(polygon)

    # # Maintain aspect ratio
    # ax.set_aspect("equal")
    # ax.set_xlim(xymins[0], xymaxs[0])
    # ax.set_ylim(xymins[1], xymaxs[1])
    # ax.set_title(f"part_{mpi.rank}")

    # # Show the plot
    # # plt.show(block=False)
    # plt.figure(fig)
    # plt.savefig(f"test_print_part_{mpi.rank}.png")


if __name__ == "__main__":
    test_mesh0()
