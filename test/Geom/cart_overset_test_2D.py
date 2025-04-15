import DNDS
import Geom
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy

def get_mesh_2D(meshFile: str):
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    mesh = Geom.UnstructuredMesh(mpi, 2)
    meshReader = Geom.UnstructuredMeshSerialRW(mesh, 0)
    assert os.path.isfile(meshFile)
    name2ID = meshReader.ReadFromCGNSSerial(meshFile)
    print(name2ID.n2id_map)
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

    return mesh, meshReader



if __name__ == "__main__":
    meshFile = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "mesh", "NACA0012_H2.cgns"
    )
    m, msr = get_mesh_2D(meshFile)
    
