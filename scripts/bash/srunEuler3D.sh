#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
cd ../build
export UCX_TLS=sm,self,ud_verbs
# export UCX_TLS=sm,self,rc,rc_verbs,ud_verbs,cm,cma,knem
# export UCX_TLS=posix,sysv,tcp,rc,cma,knem,ud,sm,self
# export HWLOC_PCI_0000_40_LOCALCPUS=
# export HWLOC_PCI_0000_44_LOCALCPUS=
# export HWLOC_PCI_0000_53_LOCALCPUS=
# export HWLOC_PCI_0000_62_LOCALCPUS=
# export HWLOC_PCI_0000_71_LOCALCPUS=
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
echo "UCX_TLS=${UCX_TLS}"
which mpirun
ls -la app/euler3D.exe
mpirun app/euler3D.exe
cd ../running
