import DNDS
import numpy as np


def get_rstart_data():
    arrayRU = DNDS.Array_d_I_I_D()
    rsize = np.linspace(3, 10, 32, dtype=np.int32)
    arrayRU.Resize(32, rsize)
    arrayRU_rstart = arrayRU.getRowStart()
    del arrayRU
    return (
        arrayRU_rstart,
        np.concatenate((np.array([0]), rsize.cumsum())),
    )


# print(dir(DNDS.Debug))
DNDS.MPI.Init_thread([])

mpi = DNDS.MPIInfo()
mpi.setWorld()

if mpi.rank == 0:
    print(f"is debugged == {DNDS.Debug.IsDebugged()}")


scalarBuf = np.zeros((), dtype=np.int64)

DNDS.MPI.Allreduce(np.array(1, dtype=np.int64), scalarBuf, "MPI_SUM", mpi)  # type: ignore

# print(f"reduced scalar {scalarBuf}")
assert scalarBuf == mpi.size

arrayR3 = DNDS.Array(int, 10)
print(f"arrayR3 is {type(arrayR3)}")

arrayR3.Resize(100)




print(arrayR3.getRowStart().shape)

arrayRU = DNDS.Array_d_I_I_D()

rsize = np.linspace(3, 10, 32, dtype=np.int32)
arrayRU.Resize(32, rsize)

print(arrayRU.Size())
assert not (np.diff(np.array(arrayRU.getRowStart())) - rsize).any()

for i in range(arrayRU.Size()):
    for j in range(arrayRU.Rowsize(i)):
        arrayRU[i, j] = i + j
arrayRUdata = np.array(arrayRU.data(), copy=False)
arrayRUdata += 1
print(arrayRUdata.shape)
for i in range(arrayRU.Size()):
    for j in range(arrayRU.Rowsize(i)):
        assert arrayRU[i, j] == i + j + 1


arrayRU_rstart = np.array(arrayRU.getRowStart(), copy=False)

(arrayRU_rstart_ret, gt) = get_rstart_data()
assert not np.any(gt - arrayRU_rstart_ret) # to see if there is memory corruption on this
print(arrayRU_rstart_ret.shape)
arrayRU_rstart_ret_np = np.array(arrayRU_rstart_ret, copy=False)
del arrayRU_rstart_ret
print(arrayRU_rstart_ret_np) # should be corrupted


print(f"{mpi.rank} / {mpi.size}, {mpi.comm():x}")


DNDS.MPI.Finalize()
