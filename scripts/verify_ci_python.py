"""Check that the installed bindings share the MPI-enabled HDF5 runtime."""

import argparse
import os
from pathlib import Path
import tempfile


def mapped_libraries(stem):
    return {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if f"/{stem}.so" in line
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5py-first", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--test-level", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()

    if args.h5py_first:
        import h5py
    from DNDSR import CFV, DNDS, EulerP, Geom
    modules = [CFV, DNDS, EulerP, Geom]
    if args.test_level >= 1:
        from DNDSR import Cauchy
        modules.append(Cauchy)
    import runtime_array_bridge_test
    if not args.h5py_first:
        import h5py
    from mpi4py import MPI

    python_root = (Path.cwd() / "python").resolve()
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(python_root), module.__file__
    bridge_root = Path(os.environ["DNDS_BUILD_DIR"]).resolve() / "test/cpp"
    assert Path(runtime_array_bridge_test.__file__).resolve().is_relative_to(bridge_root)
    assert h5py.get_config().mpi, "h5py must be built with MPI"
    hdf_libraries = mapped_libraries("libhdf5")
    packaged = python_root / "DNDSR/_lib/dndsr_external/libhdf5.so"
    assert hdf_libraries == {packaged.resolve()}, hdf_libraries
    mpi_libraries = mapped_libraries("libmpi")
    assert len(mpi_libraries) == 1, mpi_libraries
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(MPI.Get_library_version().splitlines()[0].rstrip("\0"))
        print(f"HDF5 {h5py.version.hdf5_version} from {packaged.resolve()}")
    if args.parallel:
        assert comm.size == 2
        directory = tempfile.TemporaryDirectory(prefix="dndsr-ci-h5-") if comm.rank == 0 else None
        path = comm.bcast(str(Path(directory.name) / "parallel.h5") if directory else None)
        with h5py.File(path, "w", driver="mpio", comm=comm) as checkpoint:
            values = checkpoint.create_dataset("rank_payload", (2,), dtype="i8")
            values[comm.rank] = 10 + comm.rank
        with h5py.File(path, "r", driver="mpio", comm=comm) as checkpoint:
            assert list(checkpoint["rank_payload"][:]) == [10, 11]
        comm.Barrier()
        if directory:
            directory.cleanup()


if __name__ == "__main__":
    main()
