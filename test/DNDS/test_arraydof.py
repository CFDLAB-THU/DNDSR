from DNDSR import DNDS


def test():
    import cupy as cp

    cp.cuda.Device(0).use()
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    # arr = DNDS.ArrayDOF_3_D()
    # arr.father = DNDS.ArrayEigenMatrix_3xD_3xD_D(mpi)
    # arr.son = DNDS.ArrayEigenMatrix_3xD_3xD_D(mpi)
    # arr.father.Resize(1000000, 3, 1)
    # arr.son.Resize(0, 3, 1)

    arr = DNDS.ArrayDOF_1_1()
    arr.father = DNDS.ArrayEigenMatrix_1x1_1x1_D(mpi)
    arr.son = DNDS.ArrayEigenMatrix_1x1_1x1_D(mpi)
    arr.father.Resize(1000000, 1, 1)
    arr.son.Resize(0, 1, 1)

    arr.TransAttach()

    arr.setConstant(1.0)
    print(arr.norm2() ** 2)

    arr.to_device("CUDA")
    arr.setConstant(-2.0)
    print(arr[0].tolist())
    arr.to_host()
    print(arr[0].tolist())
    arr.to_device("CUDA")
    print(arr.norm2() ** 2)
    print(arr.componentWiseNorm1())
    for _ in range(1000):
        print(arr.min())


if __name__ == "__main__":
    test()
