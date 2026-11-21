"""Small forward solver exercising the Array contract without Euler/Geom.

Native stencils, separate owned/ghost buffers, direct MPI, queued execution,
global norms, permutation/rebinding and a matching-rank parallel restart.
The independent oracle evolves the globally assembled field in NumPy.
"""
import gc
import importlib.util
import os
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R

if importlib.util.find_spec("runtime_array_bridge_test") is None:
    pytest.skip("build the native mock/bridge extension", allow_module_level=True)
import runtime_array_bridge_test as bridge


@pytest.fixture(params=["inline", "workers", "cuda"])
def backend(request):
    kind = request.param
    if kind == "cuda":
        if not R.cuda_enabled:
            pytest.skip("CPU build")
        assert bridge.cuda_aware_mpi(), "direct GPU test requires CUDA-aware MPI"
        if MPI.COMM_WORLD.size > 1:
            ids = MPI.COMM_WORLD.allgather(os.environ.get("DNDS_TEST_GPU_UUID"))
            assert None not in ids and len(set(ids)) == len(ids), "use one_rank_per_gpu.py"
    engine = R.EagerEngine(2 if kind == "workers" else 0)
    stream = engine.cuda_stream() if kind == "cuda" else engine.cpu_stream()
    device = R.Device.cuda(0) if kind == "cuda" else R.Device.host()
    yield engine, stream, device
    engine.synchronize()


@pytest.mark.parametrize("mode", [R.TransferMode.Persistent, R.TransferMode.Transient])
def test_iterate_permute_restart(backend, mode):
    engine, compute, device = backend
    comm, n, width = MPI.COMM_WORLD, 257, 5
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    channel = R.CommunicationContext(mpi)
    progress = R.MPIProgress()
    owned = R.Array(R.RowLayout.dense(n, width, 8), np.dtype("float64"), device)
    ghost = R.Array(R.RowLayout.dense(2, width, 7), owned.dtype, device)
    oracle = np.sin(np.arange(comm.size * n * width).reshape(comm.size * n, width) / 53.)
    owned[:] = oracle[comm.rank * n:(comm.rank + 1) * n]
    route = R.RowTransferPlan(channel, n, 2,
                             [R.RowRequest((comm.rank - 1) % comm.size, n - 1, 0),
                              R.RowRequest((comm.rank + 1) % comm.size, 0, 1)])
    # Both ping-pong buffers become direct MPI operands. CUDA-aware MPI alone
    # does not establish support for CUDA stream-ordered pool allocations.
    output = R.Array(owned.layout, owned.dtype, device)

    def binding(a):
        return R.PayloadPlan(route, a.layout, ghost.layout, a.element_schema()).bind(
            a, ghost, options=R.TransferOptions(mode, channel, 61))

    for step in range(9):
        transfer = binding(owned)
        # Portable MPI observes producer readiness before posting. Posting comes
        # before core launch, so that fence cannot accidentally include core work.
        operation = transfer.begin(compute.context)
        with pytest.raises(RuntimeError):
            ghost.numpy()
        with pytest.raises(RuntimeError):
            owned.write()
        done = progress.import_operation(operation)
        bridge.mock_stencil(engine, compute, owned, ghost, output, True)
        progress.drain()  # main-thread MPI progress overlaps worker/device core
        compute.wait_for(done)  # completed portable event is supported on CUDA
        bridge.mock_stencil(engine, compute, owned, ghost, output, False)
        compute.record().wait()
        engine.synchronize()
        del operation, transfer, done
        gc.collect()
        oracle = .5 * oracle + .25 * (np.roll(oracle, 1, axis=0) + np.roll(oracle, -1, axis=0))
        expected = oracle[comm.rank * n:(comm.rank + 1) * n]
        np.testing.assert_allclose(output.numpy(), expected, rtol=0, atol=2e-15)
        norm = comm.allreduce(float(np.sum(output.numpy() ** 2)))
        np.testing.assert_allclose(norm, np.sum(oracle ** 2), rtol=2e-15)
        owned, output = output, owned
        if step == 3:
            permutation = (np.arange(n) * 17) % n
            migration = R.RowTransferPlan(channel, n, n,
                                         [R.RowRequest((comm.rank + 1) % comm.size, int(i), j)
                                          for j, i in enumerate(permutation)])
            migrated = R.DynamicRowTransfer(owned, migration, owned.element_schema(), device=device,
                                           context=compute.context).commit()
            oracle = np.concatenate([oracle[((r + 1) % comm.size) * n + permutation] for r in range(comm.size)])
            retained = owned[::7, ::-2]
            old_snapshot = retained.numpy()
            owned.reset(R.RowLayout.dense(n, width, 8), device)
            owned[:] = migrated[:].numpy()  # explicit CSR-to-table materialization
            np.testing.assert_array_equal(retained.numpy(), old_snapshot)
        if step == 5:
            filename = comm.bcast(f"/tmp/dnds_mock_{os.getpid()}.h5" if comm.rank == 0 else None)
            serializer = DNDS.Serializer.SerializerH5(mpi)
            serializer.OpenFile(filename, False)
            if device.space == R.MemorySpace.CUDA:
                with pytest.raises(RuntimeError, match="staging|device"):
                    owned.write_serializer(serializer, "state", mpi)
            owned.to(R.Device.host()).write_serializer(serializer, "state", mpi)
            serializer.CloseFile()
            serializer.OpenFile(filename, True)
            restored = R.Array.read_serializer(serializer, "state", mpi).to(device)
            serializer.CloseFile()
            owned[:] = restored[:].numpy()
            comm.Barrier()
            if comm.rank == 0:
                os.remove(filename)
    engine.release_on(output, compute)
    engine.release_on(owned, compute)
    engine.synchronize()


@pytest.mark.parametrize("mode", [R.TransferMode.Persistent, R.TransferMode.Transient])
def test_direct_ragged_permutation_and_collision_reduction(backend, mode):
    _, compute, device = backend
    comm = MPI.COMM_WORLD
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    rows = 193
    lengths = [(i + comm.rank) % 9 for i in range(rows)]
    a = R.Array(R.RowLayout.padded(lengths, 11), np.dtype("int64"), device)
    for i, count in enumerate(lengths):
        a[i] = np.arange(count) + 100000 * comm.rank + 100 * i
    peer = (comm.rank + 1) % comm.size
    permutation = [(i * 17) % rows for i in range(rows)]
    route = R.RowTransferPlan(mpi, rows, rows, [R.RowRequest(peer, src, i) for i, src in enumerate(permutation)])
    b = R.Array(route.destination_layout(a.layout), a.dtype, device)
    plan = R.PayloadPlan(route, a.layout, b.layout, a.element_schema())
    binding = plan.bind(a, b, options=R.TransferOptions(mode))  # default Direct
    for _ in range(3):
        binding.begin(compute.context).wait()
        for i, src in enumerate(permutation):
            np.testing.assert_array_equal(b[i].numpy(), np.arange((src + peer) % 9) + 100000 * peer + 100 * src)
    source = R.Array(R.RowLayout.dense(3, 5, 8), np.dtype("int64"), device)
    target = R.Array(R.RowLayout.dense(1, 5, 9), source.dtype, device)
    source[:] = comm.rank + 1
    route = R.RowTransferPlan(mpi, 3, 1, [R.RowRequest(comm.rank, 0, 0), R.RowRequest(peer, 2, 0)])
    plan = R.PayloadPlan(route, source.layout, target.layout, source.element_schema())
    with pytest.raises(RuntimeError):
        plan.bind(source, target)  # colliding writes must never silently race
    reducer = R.Reducer.builtin(source.dtype, R.Reduction.Sum, R.DestinationInit.Identity)
    plan.bind(source, target, reducer, options=R.TransferOptions(mode)).begin(compute.context).wait()
    np.testing.assert_array_equal(target.numpy(), np.full((1, 5), comm.rank + peer + 2))
