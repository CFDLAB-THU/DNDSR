"""Transient MPI, dynamic shapes and eager progress using native storage."""

import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def channel():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return R.CommunicationContext(mpi)


@pytest.mark.parametrize("cuda", [False, True])
def test_retained_binding_and_slice_follow_old_mesh_snapshot_after_reset(cuda):
    if cuda and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    peer = (rank + 1) % size
    device = R.Device.cuda(0) if cuda else R.Device.host()
    context = R.ExecutionContext(device)
    ctx = channel()
    rows, width = 43, 6
    source = R.Array(R.RowLayout.dense(rows, width, 9), np.dtype("float64"), device)
    old_values = np.arange(rows * width).reshape(rows, width) + rank * 10000
    source[:] = old_values
    destination = R.Array(R.RowLayout.dense(rows, width, 8), source.dtype, device)
    retained = destination[::3, ::-2]
    route = R.RowTransferPlan(ctx, rows, rows,
                             [R.RowRequest(peer, rows - 1 - i, i) for i in range(rows)])
    plan = R.PayloadPlan(route, source.layout, destination.layout, source.element_schema())
    transport = R.TransferTransport.HostPacked if cuda else R.TransferTransport.Direct
    binding = plan.bind(source, destination, transport=transport)
    source.reset(R.RowLayout.dense(7, width), device)
    destination.reset(R.RowLayout.dense(7, width), device)
    source[:] = -17
    destination[:] = -29
    with pytest.raises(RuntimeError, match="layout"):
        plan.bind(source, destination, transport=transport)
    # The retained binding is valid for its old allocations, not the replacements.
    binding.begin(context).wait()
    expected = (np.arange(rows * width).reshape(rows, width) + peer * 10000)[::-1]
    np.testing.assert_array_equal(retained.numpy(), expected[::3, ::-2])
    np.testing.assert_array_equal(source.numpy(), np.full((7, width), -17))
    np.testing.assert_array_equal(destination.numpy(), np.full((7, width), -29))
    del binding, plan
    gc.collect()
    np.testing.assert_array_equal(retained.numpy(), expected[::3, ::-2])


@pytest.mark.parametrize("mode", [R.TransferMode.Transient, R.TransferMode.Persistent])
@pytest.mark.parametrize("transport", [R.TransferTransport.Direct, R.TransferTransport.HostPacked])
def test_shared_context_known_route_and_tag_reuse(mode, transport):
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    rows = 193
    ctx = channel()
    send, receive = [[] for _ in range(size)], [[] for _ in range(size)]
    permutation = [(i * 17) % rows for i in range(rows)]
    send[(rank - 1) % size] = permutation
    receive[(rank + 1) % size] = list(range(rows))
    route = R.RowTransferPlan.known(ctx, rows, rows, send, receive)
    source = R.Array(R.RowLayout.dense(rows, 7, 8), np.dtype("int64"))
    source[:] = np.arange(rows * 7).reshape(rows, 7) + rank * 10000
    target = R.Array(R.RowLayout.dense(rows, 7, 9), source.dtype)
    plan = R.PayloadPlan(route, source.layout, target.layout, source.element_schema())
    options = R.TransferOptions(mode, ctx, 71)
    binding = plan.bind(source, target, transport=transport, options=options)
    with pytest.raises(RuntimeError, match="tag"):
        plan.bind(source, target, transport=transport, options=options)
    for _ in range(3):
        binding.begin().wait()
        expected = np.arange(rows * 7).reshape(rows, 7)[permutation] + ((rank + 1) % size) * 10000
        np.testing.assert_array_equal(target.numpy(), expected)
    del binding
    gc.collect()
    plan.bind(source, target, transport=transport, options=options).begin().wait()


def test_sender_ordering_returns_route_and_explicit_array_permutation():
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    rows, width = 193, 5
    ctx = channel()
    send, receive = [[] for _ in range(size)], [[] for _ in range(size)]
    send_order = [(i * 17) % rows for i in range(rows)]
    send[(rank - 1) % size] = send_order
    receive[(rank + 1) % size] = list(range(rows))
    route = R.RowTransferPlan.known(ctx, rows, rows, send, receive)
    ordered_route, old_to_new = route.reorder_source_by_peer()
    assert sorted(old_to_new) == list(range(rows))
    values = np.arange(rows * width).reshape(rows, width) + rank * 10000
    permuted = np.empty_like(values)
    permuted[old_to_new] = values
    source = R.Array(R.RowLayout.dense(rows, width, 7), np.dtype("int64"))
    source[:] = permuted
    target = R.Array(R.RowLayout.dense(rows, width, 8), source.dtype)
    plan = R.PayloadPlan(ordered_route, source.layout, target.layout, source.element_schema())
    plan.bind(source, target).begin().wait()
    expected = np.arange(rows * width).reshape(rows, width)[send_order] + ((rank + 1) % size) * 10000
    np.testing.assert_array_equal(target.numpy(), expected)
    incomplete = R.RowTransferPlan.known(ctx, rows + (rank == 0), rows, send, receive)
    with pytest.raises(RuntimeError, match="each source row exactly once"):
        incomplete.reorder_source_by_peer()


@pytest.mark.parametrize("kind", ["table", "padded", "csr"])
def test_preserved_transfer_layout_defaults_and_explicit_compaction(kind):
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    lengths = [2, 2, 2] if kind == "table" else [rank % 2, 0, 2]
    shape = (R.RowLayout.dense(3, 2, 5 + rank) if kind == "table" else
             R.RowLayout.padded(lengths, 5 + rank) if kind == "padded" else R.RowLayout.ragged(lengths))
    source = R.Array(shape, np.dtype("int64"))
    for i_row, width in enumerate(lengths):
        source[i_row] = np.arange(width) + 100 * rank + 10 * i_row
    route = R.RowTransferPlan(channel(), 3, 3, [R.RowRequest((rank + 1) % size, 2 - i, i) for i in range(3)])
    assert route.destination_layout(shape).kind == shape.kind  # Existing one-argument call.
    for compact in (False, True):
        policy = R.TransferLayout.compact() if compact else R.TransferLayout()
        plan = R.PayloadPlan(route, shape, source.element_schema(), policy)
        expected_kind = R.RowLayoutKind.CSR if compact else shape.kind
        assert plan.destination_layout.kind == expected_kind
        assert route.destination_layout(shape, policy).kind == expected_kind
        if not compact and kind != "csr":
            assert plan.destination_layout.stride == shape.stride
        target = R.Array(plan.destination_layout, source.dtype)
        plan.bind(source, target).begin().wait()
        moved = R.DynamicRowTransfer(source, route, source.element_schema(), layout=policy).commit()
        assert moved.layout.kind == expected_kind
        if not compact and kind != "csr":
            assert moved.layout.stride == shape.stride
        peer = (rank + 1) % size
        for i_row in range(3):
            old_row = 2 - i_row
            width = 2 if kind == "table" or old_row == 2 else peer % 2 if old_row == 0 else 0
            expected = np.arange(width) + 100 * peer + 10 * old_row
            np.testing.assert_array_equal(target[i_row].numpy(), expected)
            np.testing.assert_array_equal(moved[i_row].numpy(), expected)


@pytest.mark.parametrize("cuda", [False, True])
def test_private_dynamic_storage_migration_and_retained_views(cuda):
    if cuda and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    device = R.Device.cuda(0) if cuda else R.Device.host()
    execution = R.ExecutionContext(device)
    ctx = channel()
    for epoch in range(3):
        rows = 67 + epoch * 2
        lengths = [(i + rank + epoch) % 9 for i in range(rows)]
        source = R.Array(R.RowLayout.padded(lengths, 11), np.dtype("int64"), device)
        for i, width in enumerate(lengths):
            source[i] = np.arange(width) + rank * 100000 + i * 100
        route = R.RowTransferPlan(ctx, rows, rows + 1,
                                 [R.RowRequest((rank + 1) % size, rows - i - 1, i) for i in range(rows)])
        operation = R.DynamicRowTransfer(source, route, source.element_schema(), device=device,
                                        transport=R.TransferTransport.HostPacked if cuda else R.TransferTransport.Direct,
                                        options=R.TransferOptions(R.TransferMode.Transient, ctx, 72), context=execution)
        with pytest.raises(RuntimeError):
            source.write()
        source.reset(R.RowLayout.dense(1, 1), device)
        result = operation.commit()
        assert result.layout.kind == R.RowLayoutKind.Padded
        assert result.layout.stride == 11
        with pytest.raises(RuntimeError, match="committed"):
            operation.commit()
        peer = (rank + 1) % size
        for i in range(rows):
            src = rows - i - 1
            width = (src + peer + epoch) % 9
            np.testing.assert_array_equal(result[i].numpy(), np.arange(width) + peer * 100000 + src * 100)
        assert result[rows].shape == (0,)
        retained = result[1]
        expected = retained.numpy()
        del result, operation
        gc.collect()
        np.testing.assert_array_equal(retained.numpy(), expected)


def test_python_reducer_lifetime_through_eager_mpi_progress():
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    ctx = channel()
    source = R.Array(R.RowLayout.dense(2, 7, 8), np.dtype("int64"))
    source[:] = rank + 1
    target = R.Array(R.RowLayout.dense(1, 7, 9), source.dtype)
    route = R.RowTransferPlan(ctx, 2, 1, [R.RowRequest(rank, 0, 0), R.RowRequest((rank + 1) % size, 1, 0)])
    reducer = R.Reducer.custom(source.element_schema(), source.dtype, "python.progress.sum", lambda a, b: a + b,
                              0, R.DestinationInit.Identity)
    binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
        source, target, reducer=reducer, options=R.TransferOptions(R.TransferMode.Transient, ctx, 73))
    with pytest.raises(RuntimeError, match="preflight"):
        binding.begin_scheduled(caller_valid=rank != 0)
    progress = R.MPIProgress()
    operation = binding.begin_scheduled()
    done = progress.import_operation(operation)
    del operation, binding, reducer
    gc.collect()
    # Event.wait releases the GIL; reducer invocation and final destruction must reacquire it.
    done.wait()
    progress.drain()
    np.testing.assert_array_equal(target.numpy(), np.full((1, 7), rank + 1 + (rank + 1) % size + 1))


@pytest.mark.parametrize("mode", ["transient", "persistent"])
def test_reducer_failure_after_posting_aborts_in_child(mode):
    """Never test a post-MPI fatal callback by aborting the pytest MPI world.

    Each child has its own singleton MPI world. The C++ distributed-failure
    fixture separately exercises one-rank failure with live remote peers.
    """
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OMPI_", "PMIX_", "PMI_"))}
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    interpreter = [sys.executable]
    if sys.flags.no_site:
        interpreter.append("-S")
    result = subprocess.run(
        [*interpreter, str(Path(__file__).with_name("runtime_reducer_failure_child.py")), mode],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "injected Python reducer reached" in result.stderr
    assert "MPI destination completion/reduction failed" in result.stderr


def test_completed_callback_does_not_retain_its_frame_and_transfer():
    def run():
        rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
        ctx = channel()
        source = R.Array(R.RowLayout.dense(31, 6, 8), np.dtype("float64"))
        target = R.Array(source.layout, source.dtype)
        route = R.RowTransferPlan(ctx, 31, 31,
                                 [R.RowRequest((rank + 1) % size, 30 - i, i) for i in range(31)])

        def combine(a, b):
            return a + b

        source.assign(np.ones((31, 6)))
        retained = weakref.ref(combine)
        reducer = R.Reducer.custom(source.element_schema(), source.dtype, "audit.python-lifetime", combine, 0,
                                  R.DestinationInit.Identity)
        binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
            source, target, reducer=reducer)
        operation = binding.begin()
        for _ in range(2):
            operation.wait()
        np.testing.assert_array_equal(target.numpy(), np.ones((31, 6)))
        return retained

    callback = run()
    gc.collect()
    assert callback() is None


@pytest.mark.mpi
@pytest.mark.parametrize("staged_cuda", [False, True])
def test_streamed_host_allreduce_orders_consumer_after_binding_release(staged_cuda):
    if staged_cuda and not R.cuda_enabled:
        pytest.skip("requires CUDA")
    shape = R.RowLayout.dense(1, 2)
    engine = R.EagerEngine(1, R.AccessTrackingContext())
    compute = engine.cuda_stream(0) if staged_cuda else engine.cpu_stream()
    device = R.Device.cuda(0) if staged_cuda else R.Device.host()
    src, observed = [R.Array(shape, np.dtype("int32"), device) for _ in range(2)]
    send, result = [R.Array(shape, src.dtype, R.Device.host()) for _ in range(2)]
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    src.assign(np.array([[rank, -rank]], dtype=np.int32))
    context = channel()
    binding = context.bind_allreduce(send, result, R.Reduction.Min)
    comm = engine.cpu_stream()
    engine.copy(compute, src, send)
    comm.wait_for(compute.record())
    reduced = binding.submit(engine, comm)
    del binding, context, send, src
    if staged_cuda:
        reduced.wait()  # Portable MPI completion must be visible before CUDA import.
    compute.wait_for(reduced)
    engine.copy(compute, result, observed)
    del result
    compute.record().wait()
    np.testing.assert_array_equal(observed.numpy(), [[0, 1 - size]])
