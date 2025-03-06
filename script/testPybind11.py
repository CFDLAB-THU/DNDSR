

import DNDSR.DNDS as DNDS


DNDS.MPI.Init_thread([])

mpi = DNDS.MPIInfo()

mpi.setWorld()

print(f"{mpi.rank} / {mpi.size}, {mpi.comm()}")

DNDS.MPI.Finalize()




