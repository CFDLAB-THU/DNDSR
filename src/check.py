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
assert scalarBuf == mpi.size

arrayR3 = DNDS.Array_d_3_3_D()

arrayR3.Resize(100)


print(arrayR3.getRowStart())

arrayRU = DNDS.Array_d_I_I_D()

rsize = np.linspace(3, 10, 32, dtype=np.int32)
arrayRU.Resize(32, rsize)

print(arrayRU.Size())
assert not (np.diff(np.array(arrayRU.getRowStart())) - rsize).any()

for i in range(arrayRU.Size()):
    for j in range(arrayRU.Rowsize(i)):
        arrayRU[i, j] = i + j
for i in range(arrayRU.Size()):
    for j in range(arrayRU.Rowsize(i)):
        assert arrayRU[i, j] == i + j

print(f"{mpi.rank} / {mpi.size}, {mpi.comm():x}")

DNDS.MPI.Finalize()
