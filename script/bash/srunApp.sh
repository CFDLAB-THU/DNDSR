#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=64
# export OMP_NUM_THREADS=32
export UCX_TLS=sm,self,ud_verbs
# export HWLOC_PCI_0000_40_LOCALCPUS=
# export HWLOC_PCI_0000_44_LOCALCPUS=
# export HWLOC_PCI_0000_53_LOCALCPUS=
# export HWLOC_PCI_0000_62_LOCALCPUS=
# export HWLOC_PCI_0000_71_LOCALCPUS=
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
# export DNDS_USE_ASYNC_ONE_BY_ONE=1
which mpirun
pwd
ls -la $1
mpirun "$@"