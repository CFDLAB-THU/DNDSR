from DNDSR import DNDS

import numpy as np
import pytest
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


@pytest.fixture
def mpi():
    world = DNDS.MPIInfo()
    world.setWorld()
    yield world


def get_rstart_data():
    arrayRU = DNDS.Array_d_I_I_Unknown()
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


def test_ParArrayPair(mpi: DNDS.MPIInfo):
    arrayIIPair = DNDS.ParArrayPair("q", "I")
    #!todo: check null status
    arrayIIPair.father = DNDS.ParArray("q", "I", init_args=(mpi,))
    arrayIIPair.son = DNDS.ParArray("q", "I", init_args=(mpi,))
    arrayIIPair.TransAttach()
    arrayIIPair.father.Resize(44)
    for iRow in range(arrayIIPair.father.Size()):
        arrayIIPair.father.ResizeRow(iRow, iRow % 4 + 2)
        for j in range(arrayIIPair.father.Rowsize(iRow)):
            arrayIIPair.father[iRow, j] = 123123
    arrayIIPair.father.Compress()
    arrayIIPair.trans.createFatherGlobalMapping()
    global_size = arrayIIPair.trans.LGlobalMapping.globalSize()
    pullIdx = range(global_size)
    arrayIIPair.trans.createGhostMapping(pullIdx)
    arrayIIPair.trans.createMPITypes()
    arrayIIPair.trans.pullOnce()
    assert arrayIIPair.son.Size() == len(pullIdx)
    assert (np.array(arrayIIPair.son.data()) == 123123).all()


def test_arrayRU(mpi: DNDS.MPIInfo):
    arrayRU = DNDS.Array_d_I_I_Unknown()

    rsize = np.linspace(3, 10, 32, dtype=np.int32)
    arrayRU.Resize(32, rsize)

    # print(arrayRU.Size())
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
        rowdata = np.array(adj[irow], copy=False)
        rowdata[:] = 3343
    adj.Compress()
    if mpi.rank == 0:
        assert not np.any(np.array(adj.data()) - 3343)

    adj_son = DNDS.ArrayAdjacency("I", init_args=(mpi,))
    transT = adj.getTrans()  # automatic transformer type get
    adj_trans = transT()
    adj_trans.setFatherSon(adj, adj_son)
    adj_trans.createFatherGlobalMapping()
    gsize = adj_trans.LGlobalMapping.globalSize()
    pullIdx = range(gsize)
    adj_trans.createGhostMapping(pullIdx)
    adj_trans.createMPITypes()
    adj_trans.pullOnce()
    assert adj_son.Size() == len(pullIdx)
    assert np.all(np.array(adj_son.data()) == 3343)


def test_ArrayEigenVector(mpi: DNDS.MPIInfo):
    arr = DNDS.ArrayEigenVector("D", init_args=(mpi,))
    arr_son = DNDS.ArrayEigenVector("D", init_args=(mpi,))
    arr.Resize(123, 5)
    val0 = np.array([1, 4, 5, 1, 3.2], dtype=np.float64)
    for irow in range(arr.Size()):
        arr[irow] = val0

    arrTrans = DNDS.ArrayTransformerFromParArray(arr)
    arrTrans.setFatherSon(arr, arr_son)
    arrTrans.createFatherGlobalMapping()
    gsize = arrTrans.LGlobalMapping.globalSize()
    pullIdx = range(gsize)
    arrTrans.createGhostMapping(pullIdx)
    arrTrans.createMPITypes()
    arrTrans.pullOnce()

    assert arr_son.Size() == len(pullIdx)
    for irow in range(arr_son.Size()):
        assert (np.array(arr_son[irow]) == val0).all()


def test_ArrayEigenMatrix(mpi: DNDS.MPIInfo):
    arr = DNDS.ArrayEigenMatrix(3, 4, init_args=(mpi,))
    arr_son = DNDS.ArrayEigenMatrix(3, 4, init_args=(mpi,))
    arr.Resize(30, 3, 4)
    mat0 = np.eye(3, 4, dtype=np.float64)
    for irow in range(arr.Size()):
        arr[irow] = mat0

    arrTrans = DNDS.ArrayTransformerFromParArray(arr)
    arrTrans.setFatherSon(arr, arr_son)
    arrTrans.createFatherGlobalMapping()
    gsize = arrTrans.LGlobalMapping.globalSize()
    pullIdx = range(gsize)
    arrTrans.createGhostMapping(pullIdx)
    arrTrans.createMPITypes()
    arrTrans.pullOnce()
    assert arr_son.Size() == len(pullIdx)
    for irow in range(arr_son.Size()):
        assert (np.array(arr_son[irow]) == mat0).all()


def test_ArrayEigenMatrixBatch(mpi: DNDS.MPIInfo):
    arr = DNDS.ArrayEigenMatrixBatch(mpi)
    arr_son = DNDS.ArrayEigenMatrixBatch(mpi)
    arr.Resize(32)

    rs = RandomState(MT19937(SeedSequence(123456789)))
    val0 = rs.rand(4, 5)
    val1 = rs.rand(2, 6)
    val2 = rs.rand(65535, 1)
    rowList = [val0, val1, val2]

    for irow in range(arr.Size()):
        rowSiz = irow % 3 + 1
        arr.InitializeWriteRow(irow, rowList[0:rowSiz])
    arr.Compress()

    arrTrans = DNDS.ArrayTransformerFromParArray(arr)
    arrTrans.setFatherSon(arr, arr_son)
    arrTrans.createFatherGlobalMapping()
    gsize = arrTrans.LGlobalMapping.globalSize()
    pullIdx = range(gsize)
    arrTrans.createGhostMapping(pullIdx)
    arrTrans.createMPITypes()
    arrTrans.pullOnce()

    assert arr_son.Size() == len(pullIdx)
    for irow in range(arr_son.Size()):
        for iMat in range(arr_son.BatchSize(irow)):
            assert (np.array(arr_son[irow, iMat]) == rowList[iMat]).all()
            # print(np.array(arr_son[irow, iMat]).tolist())


def test_ArrayEigenUniMatrixBatch(mpi: DNDS.MPIInfo):
    arr = DNDS.ArrayEigenUniMatrixBatch("D", "D", (mpi,))
    arr_son = DNDS.ArrayEigenUniMatrixBatch("D", "D", (mpi,))
    arr.Resize(32, 4, 5)
    arr_son.Resize(0, 4, 5)  #! warning: trans does not automatically do this for now!
    rs = RandomState(MT19937(SeedSequence(123456789)))
    val0 = rs.rand(4, 5)

    for irow in range(arr.Size()):
        arr.ResizeRow(irow, irow % 3 + 1)
        cDat = np.concat((val0.reshape((1, 4, 5)),) * arr.BatchSize(irow), axis=0)
        arr[irow] = cDat
    arr.Compress()

    arrTrans = DNDS.ArrayTransformerFromParArray(arr)
    arrTrans.setFatherSon(arr, arr_son)
    arrTrans.createFatherGlobalMapping()
    gsize = arrTrans.LGlobalMapping.globalSize()
    pullIdx = range(gsize)
    arrTrans.createGhostMapping(pullIdx)
    arrTrans.createMPITypes()
    arrTrans.pullOnce()

    assert arr_son.Size() == len(pullIdx)

    # print(arr_son.data().tolist())
    # print(arr_son.msize)
    # print(arr_son.BatchSize(0))
    for irow in range(arr_son.Size()):
        for iMat in range(arr_son.BatchSize(irow)):
            assert (np.array(arr_son[irow, iMat]) == val0).all()
            # print(np.array(arr_son[irow, iMat]).tolist())


if __name__ == "__main__":
    import sys

    # print(sys.argv)
    mpiC = DNDS.MPIInfo()
    mpiC.setWorld()

    if mpiC.rank == 0:
        print(f"is debugged == {DNDS.Debug.IsDebugged()}")

    test_all_reduce_scalar(mpiC)
    test_array_trans(mpiC)
    test_array_trans(mpiC, mode="left")
    test_ParArrayPair(mpiC)
    test_arrayRU(mpiC)
    test_adj(mpiC)
    test_ArrayEigenVector(mpiC)
    test_ArrayEigenMatrix(mpiC)
    test_ArrayEigenMatrixBatch(mpiC)
    test_ArrayEigenUniMatrixBatch(mpiC)

    print(f"{mpiC.rank} / {mpiC.size}, {mpiC.comm():x}")
