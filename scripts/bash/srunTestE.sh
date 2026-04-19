#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=64
cd ../build
# export UCX_TLS=ud,sm,self
# export UCX_TLS=sm,self,cm,cma,knem,sysv,posix,ud_verbs
unset UCX_TLS
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
echo ${UCX_TLS}
which mpirun
mpirun app/mpi_test.exe
cd ../running
