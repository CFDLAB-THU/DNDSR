#!/bin/bash
#SBATCH --partition=v6_384
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=96
cd ../build
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
which mpirun
mpirun app/partitionMeshSerial.exe $1 $2
cd ../running