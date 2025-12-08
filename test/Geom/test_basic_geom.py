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

    # mesh, reader, name2Id = create_mesh_from_CGNS(
    #     os.path.join(
    #         os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    #     ),
    #     mpi,
    #     2,
    # )

    mesh, reader, name2Id = create_mesh_from_CGNS(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "mesh",
            "Uniform32_Periodic.cgns",
        ),
        mpi,
        2,
        inner_process_parts=4,
        second_level_parts=4,
    )

    # mesh, reader, name2Id = create_mesh_from_CGNS(
    #     os.path.join(
    #         os.path.dirname(__file__), "..", "..", "data", "mesh", "UP3D_128.cgns"
    #     ),
    #     mpi,
    #     3,
    # )

    n2idmap = name2Id.n2id_map
    id2nmap = {k: v for v, k in n2idmap.items()}

    def name_is_wall(name: str):
        name = name.capitalize()
        if "WALL" in name:
            return True
        if "bc-4".capitalize() in name:
            return True

    def id_is_wall(id: int):
        # print(id2nmap[id])
        if id in id2nmap and name_is_wall(id2nmap[id]):
            return True
        return False

    wallDistOptions = Geom.UnstructuredMesh.WallDistOptions()
    wallDistOptions.method = 1
    wallDistOptions.subdivide_quad = 5
    wallDistOptions.verbose = 10
    wallDistOptions.wallDistExecution = 4
    mesh.BuildNodeWallDist(id_is_wall, wallDistOptions)

    meshBnd, readerBnd = create_bnd_mesh(mesh)

    mesh_bytes = mesh.getArrayBytes()
    mesh_nCell = mesh.NumCellGlobal()

    if mpi.rank == 0:
        print(f"mesh  num  cell: {mesh_nCell}")
        print(f"mesh size total: {mesh_bytes / (1024 * 1024):.4g} MB")

    mesh.coords.to_device("CUDA")
    mesh.to_device("CUDA")
    while True:
        pass

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
