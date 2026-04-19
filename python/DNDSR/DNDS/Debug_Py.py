
def MPIDebugHold():
    from mpi4py import MPI
    import time
    import debugpy

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    debugpy_port = 9000 + rank
    print(
        f"[Rank {rank}] Waiting for debugger attach on port {debugpy_port} (PID: {os.getpid()})"
    )
    debugpy.listen(("0.0.0.0", debugpy_port))
    # Wait up to 30 seconds for debugger attach
    debug_attached = 0
    while not debug_attached:
        try:
            if debugpy.is_client_connected():
                debug_attached = 1
                print(f"[Rank {rank}] Debugger attached!", flush=True)
            else:
                time.sleep(1.0)
        except TimeoutError:
            pass
        debug_attached = comm.allreduce(debug_attached, op=MPI.SUM)
