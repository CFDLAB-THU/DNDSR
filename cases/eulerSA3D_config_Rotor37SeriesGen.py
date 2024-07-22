import os.path
import json
from collections import OrderedDict
import copy
import re

directExecute = True
forceWrite = True
ignoreExist = True
waitOnExe = True #!!

baseConfName = "./eulerSA3D_config_Rotor37.json"
exePath = "../build/app/eulerSA3D.exe"
outPath = "../data/outRotor37Test1"


casePrefix = "GEN00-Rotor37_test1-O2"
orthConfigName = "eulerSA3D_config.json"
orthRunScriptName = "srunEuler.sh"

orthRunScript = r"""#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=64
export UCX_TLS=sm,self,ud_verbs
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
echo "UCX_TLS=${UCX_TLS}"
which mpirun
cp ../../../build/app/eulerSA3D.exe .
ls -la eulerSA3D.exe
mpirun eulerSA3D.exe ./eulerSA3D_config.json
rm eulerSA3D.exe
"""

class JSONWithLineCommentsDecoder(json.JSONDecoder):
    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str) -> any:
        s = '\n'.join(l if not l.lstrip().startswith('//') else '' for l in s.split('\n'))
        return super().decode(s)

fBaseConf = open(baseConfName, "r")
baseConf = json.load(fBaseConf, object_pairs_hook=OrderedDict, cls=JSONWithLineCommentsDecoder)
fBaseConf.close()
baseConf["dataIOControl"]["meshFile"] = "../../" + baseConf["dataIOControl"]["meshFile"]
baseConf["dataIOControl"]["outPltName"] = "out"

print(json.dumps(baseConf, indent=4))


psIn = 0.408461518518519
mults =  [0.8, 0.9, 1 , 1.05,1.1, 1.15, 1.2, 1.25]
vIn = 0.8



writtens = []
for mult in mults:
    caseName = casePrefix + "%g" % (mult)
    confCur = copy.deepcopy(baseConf)
    
    confCur["bcSettings"][0]["value"][4] = 0.5 * 1 * vIn**2 + psIn * mult / 0.4
    
    outDir = os.path.join(outPath, caseName)
    if not forceWrite:
        if ignoreExist and os.path.exists(outDir):
            continue
        assert not os.path.exists(outDir)
    
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, orthConfigName), "w+") as fout:
        json.dump(confCur, fout, indent=4)
    with open(os.path.join(outDir, orthRunScriptName), "w+") as fout:
        fout.write(orthRunScript)
    writtens.append(caseName)

print("Written: %d" %(len(writtens)))
print(writtens)

writtensPathAbs = [os.path.abspath(os.path.join(outPath, n)) for n in writtens]

def executeBatchCmd(paths, cmd, mw, wait):
    print(r"Force Run sbatch? (say \"%s)\" to run)" % (mw))
    
    if wait:
        dir = input()
        if dir != mw:
            return
    successJobId = []
    for n in paths:
        os.chdir(n)
        output = os.popen(cmd).read()
        print(":: " + output.strip())
        matched = re.match(r"Submitted batch job (\d+)", output)
        if matched is not None:
            successJobId.append(int(matched.group(1)))
        else:
            for id in successJobId:
                print("cancelling job %d" % (id))
                os.system("scancel %d" % (id))
            return
    else:
        print("submission complete: ")
        print(successJobId)

if directExecute:
    executeBatchCmd(writtensPathAbs, "sbatch %s" % (orthRunScriptName), "execute", waitOnExe)
   
        
        

