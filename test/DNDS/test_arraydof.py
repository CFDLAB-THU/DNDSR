from DNDSR import DNDS
import pytest


def test():
    cp = pytest.importorskip("cupy", reason="cupy not installed (requires CUDA)")

    cp.cuda.Device(0).use()
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    arr = DNDS.ArrayDOF_1_1()
    arr.father = DNDS.ArrayEigenMatrix_1x1_1x1_N(mpi)
    arr.son = DNDS.ArrayEigenMatrix_1x1_1x1_N(mpi)
    arr.father.Resize(1000000, 1, 1)
    arr.son.Resize(0, 1, 1)

    arr.TransAttach()

    arr.setConstant(1.0)
    print(arr.norm2() ** 2)

    try:
        arr.to_device("CUDA")
    except RuntimeError as e:
        if "Unknown" in str(e) or "cannot to_device" in str(e):
            pytest.skip("DNDS library built without CUDA backend support")
        raise
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
