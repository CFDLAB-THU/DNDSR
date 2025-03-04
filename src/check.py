import DNDS


DNDS.Init_thread([])

mpi = DNDS.MPIInfo()

mpi.setWorld()

print(f"{mpi.rank} / {mpi.size}, {mpi.comm()}")

DNDS.Finalize()
