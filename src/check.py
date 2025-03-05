import DNDS
import numpy as np

# print(dir(DNDS.Debug))
DNDS.MPI.Init_thread([])

mpi = DNDS.MPIInfo()
mpi.setWorld()

if mpi.rank == 0:
    print(f"is debugged == {DNDS.Debug.IsDebugged()}")


scalarBuf = np.zeros((), dtype=np.int64)

DNDS.MPI.Allreduce(np.array(1, dtype=np.int64), scalarBuf, "MPI_SUM", mpi)

# print(f"reduced scalar {scalarBuf}")
assert(scalarBuf == mpi.size)


print(f"{mpi.rank} / {mpi.size}, {mpi.comm():x}")

DNDS.MPI.Finalize()
