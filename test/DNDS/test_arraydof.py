from DNDSR import DNDS
import numpy as np
import pytest


def test():
    cp = pytest.importorskip("cupy", reason="cupy not installed (requires CUDA)")

    cp.cuda.Device(0).use()
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    arr = DNDS.ArrayDOF_1_1()
    arr.father = DNDS.ArrayEigenMatrix_1x1_1x1_N(mpi)
    arr.son = DNDS.ArrayEigenMatrix_1x1_1x1_N(mpi)
    local_rows = 17
    arr.father.Resize(local_rows, 1, 1)
    arr.son.Resize(0, 1, 1)

    arr.TransAttach()

    arr.setConstant(1.0)
    global_rows = local_rows * mpi.size
    assert arr.norm2() ** 2 == pytest.approx(global_rows)

    try:
        arr.to_device("CUDA")
    except RuntimeError as e:
        if "Unknown" in str(e) or "cannot to_device" in str(e):
            pytest.skip("DNDS library built without CUDA backend support")
        raise
    arr.setConstant(-2.0)
    arr.to_host()
    np.testing.assert_array_equal(np.asarray(arr[0]), -2.0)
    arr.to_device("CUDA")
    assert arr.norm2() ** 2 == pytest.approx(4 * global_rows)
    np.testing.assert_allclose(arr.componentWiseNorm1(), 2 * global_rows)
    np.testing.assert_allclose(arr.min(), -2.0)


if __name__ == "__main__":
    test()
