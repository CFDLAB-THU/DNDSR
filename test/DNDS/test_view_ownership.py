"""Views retain allocations, not Array objects or replacement storage."""
import gc
import weakref

import numpy as np
import pytest

from DNDSR import DNDS


def test_flat_view_retains_allocation_not_array():
    arr = DNDS.Array("d", 3)
    arr.Resize(2)
    view = arr.data()
    owner = weakref.ref(view.obj)
    array = weakref.ref(arr)
    values = np.asarray(view)
    values[:] = np.arange(6)
    assert arr[1, 2] == 5
    derived = view[1:4]
    arr.Resize(100)
    np.asarray(arr.data())[:] = 20
    values[0] = 99
    assert arr[0, 0] == 20
    del arr, view
    gc.collect()
    assert array() is None
    np.testing.assert_array_equal(values, [99, 1, 2, 3, 4, 5])
    del values
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(np.asarray(derived), [1, 2, 3])
    derived.release()
    del derived
    gc.collect()
    assert owner() is None


def test_row_starts_survive_structure_replacement():
    arr = DNDS.Array("d", "I")
    arr.Resize(2, np.array([2, 3], dtype=np.int32))
    view = arr.getRowStart()
    assert view.readonly
    values = np.asarray(view)
    with pytest.raises(ValueError):
        values.setflags(write=True)
    arr.Decompress()
    arr.ResizeRow(0, 5)
    arr.Compress()
    np.testing.assert_array_equal(values, [0, 2, 5])
    np.testing.assert_array_equal(np.asarray(arr.getRowStart()), [0, 5, 8])
    del arr, view
    gc.collect()
    np.testing.assert_array_equal(values, [0, 2, 5])


@pytest.mark.parametrize("compressed", [False, True])
def test_csr_row_views_survive_resize_and_compression(mpi, compressed):
    arr = DNDS.ArrayAdjacency("I", init_args=(mpi,))
    arr.Resize(2)
    arr.ResizeRow(0, 3)
    arr.ResizeRow(1, 2)
    arr[0] = np.array([10, 11, 12], dtype=np.int64)
    if compressed:
        arr.Compress()
    view = arr[0]
    values = np.asarray(view)
    values[1] = 19
    assert np.asarray(arr[0])[1] == 19
    if compressed:
        arr.Decompress()
    arr.ResizeRow(0, 100)
    assert np.asarray(arr[0])[1] == 19
    np.asarray(arr[0])[0] = 44
    arr.Compress()
    arr.Decompress()
    arr.Resize(0)
    del arr, view
    gc.collect()
    np.testing.assert_array_equal(values, [10, 19, 12])


@pytest.mark.parametrize("kind", ["vector", "matrix", "batch", "uniform_batch"])
@pytest.mark.parametrize("compressed", [False, True])
def test_derived_views_own_each_exported_buffer(mpi, kind, compressed):
    if kind == "vector":
        arr = DNDS.ArrayEigenVector("D", init_args=(mpi,))
        arr.Resize(2, 3)
        arr[0] = np.array([1., 2., 3.])
        def get(): return arr[0]
        def replace(): return arr.Resize(50, 3)
    elif kind == "matrix":
        arr = DNDS.ArrayEigenMatrix(3, 4, init_args=(mpi,))
        arr.Resize(2, 3, 4)
        arr[0] = np.arange(12., dtype=np.float64).reshape(3, 4)
        def get(): return arr[0]
        def replace(): return arr.Resize(50, 3, 4)
    elif kind == "batch":
        arr = DNDS.ArrayEigenMatrixBatch(mpi)
        arr.Resize(2)
        arr.InitializeWriteRow(0, [np.ones((2, 3)), np.full((3, 2), 4.)])
        if compressed:
            arr.Compress()
        # Keep an individual list item after the list itself is gone.
        def get(): return arr[0][1]
        def replace(): return arr.Decompress() if arr.IfCompressed() else None
    else:
        arr = DNDS.ArrayEigenUniMatrixBatch("D", "D", (mpi,))
        arr.Resize(2, 2, 3)
        arr.ResizeRow(0, 2)
        arr[0] = np.arange(12., dtype=np.float64).reshape(2, 2, 3)
        if compressed:
            arr.Compress()

        def get(): return arr[0]
        def replace(): return arr.Decompress() if arr.IfCompressed() else None
    view = get()
    values = np.asarray(view)
    expected = values.copy()
    values[...] += 2
    np.testing.assert_array_equal(np.asarray(get()), expected + 2)
    owner = weakref.ref(view.obj)
    array = weakref.ref(arr)
    replace()
    if kind in ("batch", "uniform_batch"):
        arr.Resize(0) if kind == "batch" else arr.Resize(0, 2, 3)
    del arr, get, replace, view
    gc.collect()
    assert array() is None
    np.testing.assert_array_equal(values, expected + 2)
    del values
    gc.collect()
    assert owner() is None


def test_empty_export_has_own_metadata():
    arr = DNDS.Array("d", 3)
    data, rows = arr.data(), arr.getRowStart()
    del arr
    gc.collect()
    assert data.shape == rows.shape == (0,)
    assert not data.readonly
    assert rows.readonly
