import DNDSR.DNDS as DNDS
import math
import numpy as np
import time
import argparse


def get_3D_grid(mpi: DNDS.MPIInfo, local_size=(32,) * 3):
    xyzdim = math.ceil(mpi.size ** (1 / 3))
    (x_dim, y_dim, z_dim) = (xyzdim,) * 3
    k = mpi.rank // (x_dim * y_dim)
    j = (mpi.rank % (x_dim * y_dim)) // x_dim
    i = mpi.rank % x_dim

    def sub2ind(ii_in, jj_in, kk_in):
        ii = ii_in % x_dim
        jj = jj_in % y_dim
        kk = kk_in % z_dim
        indC = ii + jj * x_dim + kk * (x_dim * y_dim)
        return indC % mpi.size

    neighbor_list = {
        sub2ind(i + 1, j, k): "L",
        sub2ind(i - 1, j, k): "R",
        sub2ind(i, j + 1, k): "B",
        sub2ind(i, j - 1, k): "F",
        sub2ind(i, j, k + 1): "D",
        sub2ind(i, j, k - 1): "U",
    }

    print(f"{mpi.rank}: {xyzdim}, {i}, {j}, {k}, {neighbor_list}")

    indices = np.zeros(local_size, dtype=np.int64)
    indices = np.arange(indices.size).reshape(indices.shape)
    neighbor_pull_indices_local = {}
    for rank_other, pos in neighbor_list.items():
        if pos == "L":
            neighbor_pull_indices_local[rank_other] = indices[0, :, :].reshape((-1,))
        if pos == "R":
            neighbor_pull_indices_local[rank_other] = indices[-1, :, :].reshape((-1,))
        if pos == "B":
            neighbor_pull_indices_local[rank_other] = indices[:, 0, :].reshape((-1,))
        if pos == "F":
            neighbor_pull_indices_local[rank_other] = indices[:, -1, :].reshape((-1,))
        if pos == "D":
            neighbor_pull_indices_local[rank_other] = indices[:, :, 0].reshape((-1,))
        if pos == "U":
            neighbor_pull_indices_local[rank_other] = indices[:, :, -1].reshape((-1,))
    # if mpi.rank == 0:
    #     for v in neighbor_pull_indices_local.values():
    #         print(v.shape)
    return neighbor_pull_indices_local


def bench_array_trans_3D(
    mpi: DNDS.MPIInfo, vdim=6, local_size=(32,) * 3, niter=100, see=10, warm_up=10
):
    if mpi.rank == 0:
        print(f"main data: {np.prod(local_size) * 8 * vdim} bytes")
    neighbor_pull_indices_local = get_3D_grid(mpi, local_size)
    local_size_a = np.prod(local_size)
    arr_rs = vdim if vdim <= 8 else "D"
    trans = DNDS.ArrayTransformer("d", arr_rs)
    arr = DNDS.ParArray("d", arr_rs, init_args=(mpi,))
    arr.Resize(local_size_a, vdim)
    arr_son = DNDS.ParArray("d", arr_rs, init_args=(mpi,))
    trans.setFatherSon(arr, arr_son)
    trans.createFatherGlobalMapping()
    pull = np.array([], dtype=np.int64)

    for rank_other, local_indices in neighbor_pull_indices_local.items():
        global_indices = np.copy(local_indices)
        convert = np.vectorize(lambda x: trans.LGlobalMapping(rank_other, x))
        convert(global_indices)
        pull = np.append(pull, global_indices)
    # if mpi.rank == 0:
    #     print(pull.size)

    trans.createGhostMapping(pull)
    trans.createMPITypes()
    trans.initPersistentPull()

    arrD = np.array(arr.data(), copy=False)
    arrD[:] = np.pi

    for iter in range(warm_up):
        trans.startPersistentPull()
        trans.waitPersistentPull()

    start_time = time.perf_counter()
    for iter in range(niter):
        trans.startPersistentPull()
        trans.waitPersistentPull()
        if mpi.rank == 0 and (iter + 1) % see == 0:
            print(f"Iter {iter+1} / {niter} done")

    end_time = time.perf_counter()
    iter_time = (end_time - start_time) / niter
    recv_bytes = 8 * pull.size * vdim
    recv_bw = recv_bytes / iter_time

    recv_bw_double = np.array(recv_bw, dtype=np.double)
    recv_bw_double_min = np.copy(recv_bw_double)
    recv_bw_double_max = np.copy(recv_bw_double)
    recv_bw_double_ave = np.copy(recv_bw_double)
    DNDS.MPI.Allreduce(recv_bw_double, recv_bw_double_min, "MPI_MIN", mpi)
    DNDS.MPI.Allreduce(recv_bw_double, recv_bw_double_max, "MPI_MAX", mpi)
    DNDS.MPI.Allreduce(recv_bw_double, recv_bw_double_ave, "MPI_SUM", mpi)
    recv_bw_double_ave /= mpi.size

    if mpi.rank == 0:
        print(f"rank 0 time per iter: {iter_time:.4e}")
        print(
            f"BW: {float(recv_bw_double_ave):.4e}, max {float(recv_bw_double_max):.4e}, min {float(recv_bw_double_min):.4e}"
        )

    assert not np.any(np.array(arr_son.data()) - np.pi)


if __name__ == "__main__":
    DNDS.MPI.Init_thread([])
    mpi = DNDS.MPIInfo()
    mpi.setWorld()

    parser = argparse.ArgumentParser(description="arrayTransBench")
    parser.add_argument("-s", "--size", default=32, type=int)
    parser.add_argument("-n", "--niter", default=100, type=int)
    parser.add_argument("--vdim", default=6, type=int)
    args = parser.parse_args()
    bench_array_trans_3D(
        mpi,
        vdim=args.vdim,
        niter=args.niter,
        local_size=(args.size,) * 3,
    )

    DNDS.MPI.Finalize()
