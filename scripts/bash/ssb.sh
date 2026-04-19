#!/bin/bash

if [[ $# -lt 2 ]]
then
echo "need at least 2 args, NNode and JName"
exit -1
fi

NNode=$1
JName=$2

iIter=0

PostArgs=""

for str in "$@"
do
iIter=$((iIter+1))
if [[ $iIter -gt 2 ]]
then
# echo $str
PostArgs="$PostArgs $str"
fi
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

runCmd="sbatch -N$NNode -J $JName -o $JName.txt ${SCRIPT_DIR}/srunApp.sh $PostArgs"
# echo $runCmd
$runCmd


# sbatch -N10 -J DLR_D_T1 -o DLR_D_T1.txt srunApp.sh ../build/app/eulerSA3D.exe ../cases/eulerSA3D_config_DLRF6.json