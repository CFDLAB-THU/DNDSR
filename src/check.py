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


def test_all_reduce_scalar(mpi: DNDS.MPIInfo):
    scalarBuf = np.zeros((), dtype=np.int64)
    DNDS.MPI.Allreduce(np.array(1, dtype=np.int64), scalarBuf, "MPI_SUM", mpi)  # type: ignore
    # print(f"reduced scalar {scalarBuf}")
    assert scalarBuf == mpi.size


def test_array_trans(mpi: DNDS.MPIInfo, mode: str = "global"):
    arrayR3 = DNDS.ParArray("d", 3, init_args=(mpi,))
    if mpi.rank == 0:
        print(f"arrayR3 is {type(arrayR3)}")
        print(arrayR3.getRowStart().shape)
    arrayR3.Resize(100)
    d = arrayR3.data()
    np.array(d, copy=False)[:] = 1.335

    arrayR3son = DNDS.ParArray("d", 3, init_args=(mpi,))
    arrayR3Trans = DNDS.ArrayTransformer("d", 3)
    arrayR3Trans.setFatherSon(arrayR3, arrayR3son)
    arrayR3Trans.createFatherGlobalMapping()
    global_size = arrayR3.getLGlobalMapping().globalSize()
    pullIdx = range(global_size)
    if mode == "left":
        pullIdx = np.array([0, 1, 2, 3], dtype=np.int64)
        rank_pull = (mpi.rank - 1) % mpi.size
        np.vectorize(lambda x: arrayR3.getLGlobalMapping()(rank_pull, x))(pullIdx)

    arrayR3Trans.createGhostMapping(pullIdx)
    arrayR3Trans.createMPITypes()
    arrayR3Trans.pullOnce()
    if mpi.rank == 0:
        print(f"arrayR3son.size is {arrayR3son.Size()}")
    assert arrayR3son.Size() == len(pullIdx)
    assert np.all(np.array(arrayR3son.data()) == 1.335)

    arrayR3son_1 = DNDS.ParArray("d", 3, init_args=(mpi,))
    arrayR3Trans_1 = DNDS.ArrayTransformer("d", 3)
    arrayR3Trans_1.setFatherSon(arrayR3, arrayR3son_1)
    arrayR3Trans_1.BorrowGGIndexing(arrayR3Trans)
    arrayR3Trans_1.createMPITypes()
    arrayR3Trans_1.pullOnce()

    assert arrayR3son_1.Size() == len(pullIdx)
    assert np.all(np.array(arrayR3son_1.data()) == 1.335)


def test_arrayRU(mpi: DNDS.MPIInfo):
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
    if mpi.rank == 0:
        print(arrayRUdata.shape)
    for i in range(arrayRU.Size()):
        for j in range(arrayRU.Rowsize(i)):
            assert arrayRU[i, j] == i + j + 1

    (arrayRU_rstart_ret, gt) = get_rstart_data()
    assert not np.any(
        gt - arrayRU_rstart_ret
    )  # to see if there is memory corruption on this
    if mpi.rank == 0:
        print(arrayRU_rstart_ret.shape)
    arrayRU_rstart_ret_np = np.array(arrayRU_rstart_ret, copy=False)
    del arrayRU_rstart_ret
    if mpi.rank == 0:
        print(
            f"if corrupted: { np.any(gt - arrayRU_rstart_ret_np)}"
        )  # should be corrupted


def test_adj(mpi: DNDS.MPIInfo):
    adj = DNDS.ArrayAdjacency("I", init_args=(mpi,))
    adj.Resize(100)
    for irow in range(adj.Size()):
        adj.ResizeRow(irow, irow % 3 + 1)
        rowdata = np.array(adj[(irow,)], copy=False)
        rowdata[:] = 3343
    adj.Compress()
    if mpi.rank == 0:
        assert not np.any(np.array(adj.data()) - 3343)


# print(dir(DNDS.Debug))
DNDS.MPI.Init_thread([])

mpi = DNDS.MPIInfo()
mpi.setWorld()

if mpi.rank == 0:
    print(f"is debugged == {DNDS.Debug.IsDebugged()}")

test_all_reduce_scalar(mpi)
test_array_trans(mpi)
test_array_trans(mpi, mode="left")
test_arrayRU(mpi)
test_adj(mpi)

print(f"{mpi.rank} / {mpi.size}, {mpi.comm():x}")


DNDS.MPI.Finalize()
