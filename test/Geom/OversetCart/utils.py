import DNDS, Geom, os, sys
import mpi4py.MPI as MPI
import numpy as np


def get_mesh_2D(meshFile: str, mpi: DNDS.MPIInfo):

    mesh = Geom.UnstructuredMesh(mpi, 2)
    meshReader = Geom.UnstructuredMeshSerialRW(mesh, 0)
    assert os.path.isfile(meshFile)
    name2ID = meshReader.ReadFromCGNSSerial(meshFile)
    # print(name2ID.n2id_map)
    meshReader.Deduplicate1to1Periodic(1e-9)
    meshReader.BuildCell2Cell()
    meshReader.MeshPartitionCell2Cell({"metisUfactor": 5})
    meshReader.PartitionReorderToMeshCell2Cell()

    mesh.RecoverNode2CellAndNode2Bnd()
    mesh.RecoverCell2CellAndBnd2Cell()
    mesh.BuildGhostPrimary()
    mesh.AdjGlobal2LocalPrimary()
    mesh.InterpolateFace()
    mesh.AssertOnFaces()

    return mesh, meshReader, name2ID


def get_mpi4py_comm_from_MPIInfo(mpi: DNDS.MPIInfo):
    return MPI.Comm.fromhandle(mpi.comm())  # handling raw MPI_Comm handle (pointer)

t_travelling_cell_pack = tuple[Geom.Elem.ElemType, int, int, list[int], np.ndarray]

def pack_travelling_cell(
    cellType: Geom.Elem.ElemType,
    cellZone: int,
    iCell: int,
    cell2nodeRow: list[int],
    coords: np.ndarray,
):
    return (cellType, cellZone, iCell, cell2nodeRow, coords)
