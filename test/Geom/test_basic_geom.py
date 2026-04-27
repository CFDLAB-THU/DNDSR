# import DNDS
# import Geom


from DNDSR import DNDS as DNDS
from DNDSR import Geom as Geom
from DNDSR.Geom.utils import create_mesh_from_CGNS, create_bnd_mesh, read_mesh, prepare_mesh
import os
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

    try:
        mesh.coords.to_device("CUDA")
        mesh.to_device("CUDA")
    except RuntimeError:
        pass  # CUDA not available, skip device transfer
    # while True:
    #     pass

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


def test_multi_layer_ghost():
    """Test multi-layer ghost cell creation with nGhostLayers=1,2,3.

    Verifies:
    1. Ghost cell/node/bnd counts increase monotonically with layer count.
    2. For nLayers=2, every layer-1 ghost cell's cell2cell neighbors are
       in the local+ghost set (the defining property of 2-layer ghosts).
    3. The full pipeline (read + prepare) works for each layer count.
    """
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    mesh_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh",
        "Uniform32_Periodic.cgns",
    )

    ghost_counts = {}  # nLayers -> (nCellGhost, nNodeGhost, nBndGhost)

    for nLayers in [1, 2, 3]:
        mesh = Geom.UnstructuredMesh(mpi, 2)
        reader = Geom.UnstructuredMeshSerialRW(mesh, 0)
        reader.ReadFromCGNSSerial(mesh_file)
        reader.Deduplicate1to1Periodic(1e-9)
        reader.BuildCell2Cell()
        reader.MeshPartitionCell2Cell({
            "metisType": "KWAY",
            "metisUfactor": 5,
            "metisSeed": 0,
            "metisNcuts": 3,
        })
        reader.PartitionReorderToMeshCell2Cell()

        mesh.RecoverNode2CellAndNode2Bnd()
        mesh.RecoverCell2CellAndBnd2Cell()
        mesh.BuildGhostPrimary(nLayers)
        mesh.AdjGlobal2LocalPrimary()
        mesh.AdjGlobal2LocalN2CB()

        nCellOwned = mesh.NumCell()
        nCellGhost = mesh.NumCellGhost()
        nNodeGhost = mesh.NumNodeGhost()
        nBndGhost = mesh.NumBndGhost()

        ghost_counts[nLayers] = (nCellGhost, nNodeGhost, nBndGhost)

        if mpi.rank == 0:
            print(f"  nLayers={nLayers}: owned={nCellOwned}, "
                  f"cellGhost={nCellGhost}, nodeGhost={nNodeGhost}, "
                  f"bndGhost={nBndGhost}")

        # For nLayers >= 2: verify that every owned cell's cell2cell
        # neighbor's cell2cell neighbors are all in the local+ghost set.
        # This is the defining property of 2-layer ghosts.
        if nLayers >= 2:
            nCellProc = mesh.NumCellProc()  # father + son
            for iCell in range(nCellOwned):
                for iNeighbor in mesh.cell2cell[iCell].tolist():
                    if iNeighbor < 0:
                        continue  # not-found encoded, skip
                    assert iNeighbor < nCellProc, (
                        f"Layer-1 neighbor {iNeighbor} of owned cell {iCell} "
                        f"is out of range [0, {nCellProc})")
                    # Check that iNeighbor's own neighbors are also resolvable
                    for iNeighbor2 in mesh.cell2cell[iNeighbor].tolist():
                        if iNeighbor2 < 0:
                            continue
                        assert iNeighbor2 < nCellProc, (
                            f"Layer-2 neighbor {iNeighbor2} of cell {iNeighbor} "
                            f"(neighbor of owned cell {iCell}) is out of range "
                            f"[0, {nCellProc})")

        # Run the full prepare pipeline to make sure it doesn't crash.
        prepare_mesh(mesh, reader, build_serial_out=False)

        del mesh, reader

    # Verify monotonic increase of ghost counts.
    for key in ["cell", "node", "bnd"]:
        idx = {"cell": 0, "node": 1, "bnd": 2}[key]
        counts = [ghost_counts[n][idx] for n in [1, 2, 3]]
        for i in range(len(counts) - 1):
            assert counts[i + 1] >= counts[i], (
                f"{key} ghost count should be monotonically non-decreasing: "
                f"nLayers={i + 1}->{i + 2}: {counts[i]}->{counts[i + 1]}")
        if mpi.rank == 0:
            print(f"  {key} ghost counts: {counts}")

    # With mpi.size >= 2, ghost counts should strictly increase.
    if mpi.size >= 2:
        for key in ["cell", "node"]:
            idx = {"cell": 0, "node": 1}[key]
            assert ghost_counts[2][idx] > ghost_counts[1][idx], (
                f"{key} ghost count should increase from 1 to 2 layers "
                f"with {mpi.size} ranks: "
                f"{ghost_counts[1][idx]} vs {ghost_counts[2][idx]}")


if __name__ == "__main__":
    test_mesh0()
    test_multi_layer_ghost()
