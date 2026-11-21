"""Native DOF algebra shares Runtime storage; no Python numerical substitute."""

import gc
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R

A = R.Algebra


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
@pytest.mark.parametrize("width", [5, 6, 7, 114])
def test_native_owned_operations_share_python_values_and_leave_ghosts(dtype, width):
    space = A.DofSpace(19, 3)
    x = A.DofField(space, width, dtype, width + 3)
    y = A.DofField(space, width, dtype, width + 1)
    values = (1 + np.arange(19)[:, None] * 2 + np.arange(width)[None, :] / 4).astype(dtype)
    x.owned.assign(values)
    x.ghost.assign(np.full((3, width), -77, dtype=dtype))
    y.ghost.assign(np.full((3, width), -99, dtype=dtype))
    retained = x.owned
    A.fill(y, 3)
    A.axpy(y, 2, x)
    np.testing.assert_array_equal(y.owned.numpy(), 3 + 2 * values)
    A.linear_combination(y, 0.5, y, -1, x)
    np.testing.assert_array_equal(y.owned.numpy(), np.full((19, width), 1.5, dtype=dtype))
    A.scale(x, 2)
    np.testing.assert_array_equal(retained.numpy(), 2 * values)
    np.testing.assert_array_equal(x.ghost.numpy(), np.full((3, width), -77, dtype=dtype))
    np.testing.assert_array_equal(y.ghost.numpy(), np.full((3, width), -99, dtype=dtype))
    assert A.dot_local(y, y) == 19 * width * 2.25
    assert A.norm_local(y) == pytest.approx(1.5 * np.sqrt(19 * width))
    np.testing.assert_array_equal(A.component_sum_local(y), np.full(width, 19 * 1.5))
    alias = type(x)(space, retained, x.ghost)
    assert alias.owned.identity == retained.identity
    A.fill(alias, 7)
    np.testing.assert_array_equal(x.owned.numpy(), np.full((19, width), 7, dtype=dtype))
    A.copy(y, x)
    assert y.owned.identity != x.owned.identity
    A.scale(x, 3)
    np.testing.assert_array_equal(y.owned.numpy(), np.full((19, width), 7, dtype=dtype))


@pytest.mark.parametrize("device", [R.Device.host(), R.Device.cuda(0)])
def test_eager_algebra_cross_stream_results_retain_native_operands(device):
    if device.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("CPU profile has no CUDA algebra")
    space = A.DofSpace(257, 2)
    x, y = (A.DofField(space, 7, np.dtype("float64"), 10, device) for _ in range(2))
    engine = R.EagerEngine(2)
    producer = engine.cpu_stream() if device.space == R.MemorySpace.Host else engine.cuda_stream(0)
    consumer = engine.cpu_stream() if device.space == R.MemorySpace.Host else engine.cuda_stream(0)
    A.fill_async(engine, producer, x, 4)
    consumer.wait_for(producer.record())
    A.copy_async(engine, consumer, y, x)
    A.axpy_async(engine, consumer, y, 0.5, x)
    del x
    gc.collect()
    result = A.norm_local_async(engine, consumer, y)
    assert result.get() == pytest.approx(6 * np.sqrt(257 * 7))
    engine.synchronize()
    np.testing.assert_array_equal(y.owned.numpy(), np.full((257, 7), 6.))
    varied = (-3.25 + (np.arange(257)[:, None] % 17) / 2 - np.arange(7)[None, :] / 4)
    y.owned.assign(varied)
    np.testing.assert_array_equal(A.component_min_local_async(engine, consumer, y).get(), varied.min(axis=0))
    np.testing.assert_array_equal(A.component_max_local_async(engine, consumer, y).get(), varied.max(axis=0))
    np.testing.assert_array_equal(A.component_sum_local_async(engine, consumer, y).get(), varied.sum(axis=0))
    A.fill(y, 2)
    np.testing.assert_array_equal(y.owned.numpy(), np.full((257, 7), 2.))


def test_global_reductions_count_uneven_owned_rows_and_reject_other_spaces_collectively():
    comm = MPI.COMM_WORLD
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    rows = 0 if comm.size > 1 and comm.rank + 1 == comm.size else comm.rank + 2
    space = A.DofSpace.distributed(mpi, rows, 3)
    x, y = A.DofField(space, 6), A.DofField(space, 6)
    A.fill(x, comm.rank + 1)
    A.fill(y, 2)
    counts = [0 if comm.size > 1 and rank + 1 == comm.size else rank + 2 for rank in range(comm.size)]
    assert A.dot_global(x, y) == sum(count * 6 * (rank + 1) * 2 for rank, count in enumerate(counts))
    assert A.norm_global(x) == pytest.approx(np.sqrt(sum(count * 6 * (rank + 1) ** 2 for rank, count in enumerate(counts))))
    np.testing.assert_array_equal(A.component_sum_global(x), np.full(6, sum(count * (rank + 1) for rank, count in enumerate(counts))))
    other = A.DofSpace(rows, 3) if comm.rank == 0 else space
    wrong = A.DofField(other, 6)
    with pytest.raises(RuntimeError, match="distribution|rank"):
        A.dot_global(x, wrong)
    assert A.dot_global(x, y) == sum(count * 6 * (rank + 1) * 2 for rank, count in enumerate(counts))


def test_field_import_rejects_wrong_dtype_variable_layout_and_space():
    space = A.DofSpace(2)
    x = A.DoubleDofField(space, 5)
    with pytest.raises(RuntimeError, match="distribution"):
        A.copy(x, A.DoubleDofField(A.DofSpace(2), 5))
    with pytest.raises(TypeError):
        A.DoubleDofField(space, R.Array(R.RowLayout.dense(2, 5), np.dtype("int64")), x.ghost)
    with pytest.raises(RuntimeError):
        A.DoubleDofField(space, R.Array(R.RowLayout.ragged([5, 5])), x.ghost)
    with pytest.raises(RuntimeError):
        A.DofField(space, 5, np.dtype("int64"))


def test_cuda_global_reductions_wait_for_local_device_results_before_mpi():
    if not R.cuda_enabled:
        pytest.skip("CPU profile has no CUDA algebra")
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    comm = MPI.COMM_WORLD
    space = A.DofSpace.distributed(mpi, comm.rank + 2, 3)
    x = A.DofField(space, 6, np.dtype("float32"), 9, R.Device.cuda(0))
    y = A.DofField(space, 6, np.dtype("float32"), 8, R.Device.cuda(0))
    engine = R.EagerEngine()
    stream = engine.cuda_stream(0)
    A.fill_async(engine, stream, x, comm.rank + 1)
    A.fill_async(engine, stream, y, 2)
    # No preceding host wait: global algebra must obtain the local CUDA scalar
    # before collective MPI, on this MPI-permitted calling thread.
    expected_dot = sum((rank + 2) * 6 * (rank + 1) * 2 for rank in range(comm.size))
    expected_square = sum((rank + 2) * 6 * (rank + 1) ** 2 for rank in range(comm.size))
    assert A.dot_global(engine, stream, x, y) == expected_dot
    assert A.norm_global(engine, stream, x) == pytest.approx(np.sqrt(expected_square))
    expected_sum = sum((rank + 2) * (rank + 1) for rank in range(comm.size))
    np.testing.assert_array_equal(A.component_sum_global(engine, stream, x), np.full(6, expected_sum))
    np.testing.assert_array_equal(A.component_min_global(engine, stream, x), np.ones(6))
    np.testing.assert_array_equal(A.component_max_global(engine, stream, x), np.full(6, comm.size))
    engine.synchronize()


@pytest.mark.parametrize("device", [R.Device.host(), R.Device.cuda(0)])
@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
def test_physical_tables_share_native_fields_and_async_mixed_layout_algebra(device, dtype):
    if device.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("CPU profile has no CUDA algebra")
    width, rows = 37, 259
    space = A.DofSpace(rows, 3)
    cls = A.FloatDofField if dtype == np.dtype("float32") else A.DoubleDofField
    x = cls(space, R.RowLayout.col_major(0, width, 32, 1), device)
    y = cls(space, R.RowLayout.tiled_col_major(0, width, 16), device)
    values = (1 + np.arange(rows)[:, None] / 4 + np.arange(width)[None, :] / 8).astype(dtype)
    x.owned.assign(values)
    y.ghost.assign(np.full((3, width), -99, dtype=dtype))
    alias = cls(space, x.owned, x.ghost)
    assert alias.owned.identity == x.owned.identity
    engine = R.EagerEngine(2)
    stream = engine.cuda_stream(0) if device.space == R.MemorySpace.CUDA else engine.cpu_stream()
    A.copy_async(engine, stream, y, alias)
    A.axpy_async(engine, stream, y, 2, x)
    expected = values * 3
    np.testing.assert_array_equal(A.component_sum_local_async(engine, stream, y).get(), expected.sum(axis=0, dtype=np.float64))
    engine.synchronize()
    np.testing.assert_array_equal(y.owned.numpy(), expected)
    np.testing.assert_array_equal(y.ghost.numpy(), np.full((3, width), -99, dtype=dtype))
    np.testing.assert_array_equal(x.owned.numpy(), values)
