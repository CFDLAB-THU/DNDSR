"""Optional zero-copy interoperability; no Torch dependency in DNDS itself.

DNDS_REQUIRE_TORCH=1 turns missing Torch into failure for the interop job.
Finish foreign CUDA uses and join other streams before the last alias dies.
"""
import gc
import importlib.util
import os
import numpy as np
import pytest
from DNDSR.DNDS import Runtime as R


pytestmark = pytest.mark.level2


@pytest.fixture
def torch():
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("DNDS_REQUIRE_TORCH"):
            pytest.fail("Torch is required by this validation run")
        pytest.skip("optional Torch integration")
    import torch
    return torch


@pytest.fixture(params=[False, True])
def device(request, torch):
    if request.param and not R.cuda_enabled:
        pytest.skip("CPU DNDS build")
    if request.param:
        assert torch.cuda.is_available()
    return R.Device.cuda(0) if request.param else R.Device.host()


@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_export_pointer_mutation_alias_lifetime_and_generation(torch, device, dtype):
    a = R.Array(R.RowLayout.dense(31, 5, 8), np.dtype(dtype), device)
    a[:] = np.arange(155).reshape(31, 5) % 97
    with a.read() as access:
        address = access.unsafe_address()
    version = a.version
    tensor = torch.from_dlpack(a)
    assert tensor.data_ptr() == address and tensor.stride() == (8, 1)
    tensor.fill_(2)  # Torch does not implement every arithmetic op for uint16/32/64.
    retained = tensor[::3, 1::2]
    del tensor
    gc.collect()
    with pytest.raises(RuntimeError):
        a.numpy()
    with pytest.raises(RuntimeError):
        a.write()
    retained.fill_(5)
    del retained
    gc.collect()
    expected = np.full((31, 5), 2)
    expected[::3, 1::2] = 5
    np.testing.assert_array_equal(a.numpy(), expected)
    assert a.version == version + 1


def test_export_survives_reset_and_native_owner(torch, device):
    import runtime_array_bridge_test as bridge
    owner = bridge.NativeOwner(device)
    a = owner.state
    a[:] = 11
    tensor = torch.from_dlpack(a)
    a.reset(R.RowLayout.dense(2, 5), device)
    del owner
    gc.collect()
    tensor.mul_(4)
    np.testing.assert_array_equal(tensor.cpu().numpy(), np.full((3, 5), 44))
    np.testing.assert_array_equal(a.numpy(), np.zeros((2, 5)))


def test_import_retains_producer_and_shares_with_cpp(torch, device):
    import runtime_array_bridge_test as bridge
    native = "cuda:0" if device.space == R.MemorySpace.CUDA else "cpu"
    tensor = torch.arange(195, dtype=torch.float64, device=native).reshape(39, 5)
    a = R.from_dlpack(tensor)
    with a.read(stream=0) as lease:
        assert lease.unsafe_address() == tensor.data_ptr()
    bridge.scale(a, 2)
    np.testing.assert_array_equal(tensor.cpu().numpy(), np.arange(195).reshape(39, 5) * 2)
    retained = a[::3, ::-2]
    del tensor, a
    gc.collect()
    np.testing.assert_array_equal(retained.numpy(), (np.arange(195).reshape(39, 5) * 2)[::3, ::-2])


def test_import_owner_survives_queued_kernel_copy_and_release(torch, device):
    import runtime_array_bridge_test as bridge
    cuda = device.space == R.MemorySpace.CUDA
    engine = R.EagerEngine(2)
    stream = engine.cuda_stream() if cuda else engine.cpu_stream()
    tensor = torch.full((4099, 5), 7., dtype=torch.float64, device="cuda:0" if cuda else "cpu")
    a = R.from_dlpack(tensor)
    del tensor
    output = engine.allocate_like(stream, a)
    gate = bridge.hold(engine, stream)
    try:
        bridge.scale_async(engine, stream, a, 3)
        engine.copy(stream, a, output)
        engine.release_on(a, stream)
        assert a.shape == (0, 0)
        with pytest.raises(RuntimeError):
            output.numpy()
    finally:
        gate.open()
    stream.record().wait()
    engine.synchronize()
    np.testing.assert_array_equal(output.numpy(), np.full((4099, 5), 21.))


def test_cuda_nondefault_producer_consumer_and_explicit_join(torch):
    if not R.cuda_enabled:
        pytest.skip("CPU DNDS build")
    producer, consumer, side = (torch.cuda.Stream() for _ in range(3))
    with torch.cuda.stream(producer):
        tensor = torch.full((4099, 5), 7., dtype=torch.float64, device="cuda")
        tensor.mul_(3)
        a = R.from_dlpack(tensor)  # producer establishes readiness on legacy stream
    del tensor
    with torch.cuda.stream(consumer):
        foreign = torch.from_dlpack(a)
        foreign.add_(4)
    side.wait_stream(consumer)
    with torch.cuda.stream(side):
        foreign.mul_(2)
    consumer.wait_stream(side)  # caller joins independent foreign use before release
    del foreign
    gc.collect()
    np.testing.assert_array_equal(a.numpy(), np.full((4099, 5), 50.))


def test_numpy_export_capsule_once_and_failure_cleanup():
    a = R.Array(R.RowLayout.dense(13, 7), np.dtype("int64"))
    a[:] = 9
    exported = np.from_dlpack(a)
    assert not exported.flags.writeable
    with pytest.raises(RuntimeError):
        a[0, 0]
    del exported
    gc.collect()
    assert a[0, 0] == 9
    capsule = a.__dlpack__()
    del capsule  # unconsumed capsule must release the lease too
    gc.collect()
    assert a[0, 0] == 9
    with pytest.raises(ValueError):
        a.__dlpack__(copy=True)
    with pytest.raises(ValueError):
        a.__dlpack__(stream=1)
    assert a[0, 0] == 9


def test_engine_destructor_drains_foreign_owner_without_holding_gil():
    import runtime_array_bridge_test as bridge
    engine = R.EagerEngine(1)
    stream = engine.cpu_stream()
    producer = np.arange(95, dtype="float64").reshape(19, 5)
    a = R.from_dlpack(producer)
    del producer
    bridge.delay_cpu(engine, stream)
    bridge.scale_async(engine, stream, a, 2)
    engine.release_on(a, stream)
    del a, engine  # last foreign-owner decrement runs on the worker during drain


def test_reject_layouts_readonly_and_unsupported_dtype(torch):
    for layout in [R.RowLayout.ragged([1, 2]), R.RowLayout.padded([1, 2], 3)]:
        with pytest.raises(RuntimeError, match="table"):
            torch.from_dlpack(R.Array(layout))
    with pytest.raises(RuntimeError, match="numeric"):
        torch.from_dlpack(R.Array(R.RowLayout.dense(2, 3), np.dtype("V16")))
    source = torch.arange(48, dtype=torch.float64).reshape(6, 8)
    for unsupported in [source.t(), source[:, ::2], source[:, :5], source.reshape(2, 3, 8)]:
        with pytest.raises(RuntimeError, match="contiguous|shape"):
            R.from_dlpack(unsupported)
    readonly = np.arange(10, dtype="float64")
    readonly.flags.writeable = False
    with pytest.raises(BufferError):
        R.from_dlpack(readonly)


def test_empty_vector_offset_import_and_roundtrip(torch):
    for source in [torch.empty((0, 5), dtype=torch.float64), torch.arange(17, dtype=torch.int64)[3:14]]:
        a = R.from_dlpack(source)
        b = torch.from_dlpack(a)
        assert b.data_ptr() == source.data_ptr()
        np.testing.assert_array_equal(b.numpy().reshape(-1), source.numpy().reshape(-1))
        del b
        gc.collect()
        assert a.valid


def test_capsule_single_consumption_and_runtime_roundtrip(torch):
    a = R.Array(R.RowLayout.dense(17, 5))
    a[:] = 23
    capsule = a.__dlpack__()
    tensor = torch.from_dlpack(capsule)
    with pytest.raises(RuntimeError):
        torch.from_dlpack(capsule)
    del tensor, capsule
    gc.collect()
    b = R.from_dlpack(a)
    b[2, 3] = 71
    with pytest.raises(RuntimeError):
        a[2, 3]
    del b
    gc.collect()
    assert a[2, 3] == 71

@pytest.mark.parametrize("layout", ["col", "tiled"])
def test_physical_table_exports_are_truthful_and_retain_lease(torch, device, layout):
    shape = R.RowLayout.col_major(33, 5, 32, 1) if layout == "col" else R.RowLayout.tiled_col_major(33, 5, 8)
    a = R.Array(shape, np.dtype("float64"), device, alignment_bytes=4096)
    expected = np.arange(165).reshape(33, 5).astype(np.float64)
    a[:] = expected
    with a.read() as access:
        address = access.unsafe_address()
    assert address % 4096 == 0
    tensor = torch.from_dlpack(a)
    assert tensor.data_ptr() == address
    if layout == "col":
        assert tensor.shape == (33, 5) and tensor.stride() == (1, 64)
        np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
        tensor.add_(7)
        retained = tensor[::2, :]
    else:
        assert tensor.shape == (5, 5, 8) and tensor.stride() == (40, 8, 1)
        logical = tensor.permute(0, 2, 1).reshape(40, 5)[:33]
        np.testing.assert_array_equal(logical.cpu().numpy(), expected)
        tensor.add_(7)
        retained = tensor[4:5]
    del tensor
    gc.collect()
    with pytest.raises(RuntimeError):
        a.numpy()
    del retained
    gc.collect()
    np.testing.assert_array_equal(a.numpy(), expected + 7)
    view = a[::2, ::-1]
    a.reset(R.RowLayout.dense(1, 5), device)
    np.testing.assert_array_equal(view.numpy(), (expected + 7)[::2, ::-1])
