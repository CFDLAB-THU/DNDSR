import DNDS
import Geom
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sympy


def sum_partition_greedy(arr: np.ndarray, n_part: int = 2):
    """Partitions numbers into M subsets using Largest-First Decreasing (LFD)."""
    numbers = sorted(
        [(arr.flat[i], i) for i in range(arr.size)], reverse=True, key=lambda v: v[0]
    )  # Sort largest first
    sums = np.array([0] * n_part)  # Track partition sums

    partitions = np.zeros_like(arr, dtype=np.int64)

    for num, i in numbers:
        idx = np.argmin(sums)  # Find partition with smallest sum
        sums[idx] += num  # Update sum
        partitions.flat[i] = idx

    return partitions


def int_factor_divide(N: int, n_part: int = 2):
    factors = sympy.factorint(N)
    factor_list = []
    for f, p in factors.items():
        factor_list += [f] * p
    factor_list = np.array(factor_list, dtype=np.int64)
    assert factor_list.prod() == N
    partition = sum_partition_greedy(
        np.log2((factor_list * 2).astype(np.float64)), n_part
    )
    prods = np.array([np.prod(factor_list[partition == i]) for i in range(n_part)])
    assert prods.prod() == N
    return prods


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

    for i in np.arange(2, 201) * 64:
        prods = int_factor_divide(i)
        diffv = prods.max() - prods.min()
        if diffv > np.sqrt(i) * 0:
            print(f"{i:6}: {str(prods):32}: {diffv}")
