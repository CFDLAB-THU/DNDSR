from DNDSR import DNDS


def test():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    arr = DNDS.ArrayDOF_3_D()
    arr.father = DNDS.ArrayEigenMatrix_3xD_3xD_D(mpi)
    arr.son = DNDS.ArrayEigenMatrix_3xD_3xD_D(mpi)
    arr.father.Resize(1000, 3, 1)
    arr.son.Resize(0, 3, 1)
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


if __name__ == "__main__":
    test()
