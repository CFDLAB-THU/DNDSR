import os.path
import json
from collections import OrderedDict
import copy
import re

directExecute = True
forceWrite = True
ignoreExist = True
waitOnExe = True #!!

baseConfName = "./euler_config_vorstreetBenchTmp.json"
exePath = "../build/app/euler.exe"
outPath = "../data/outCylinder1"

dtBase = 0.125e-1
stepBase = 800
zeroRecForStepsBase = 0
casePrefix = "GEN01_CylinderB1_RE1200_M01_C"
orthConfigName = "euler_config.json"
orthRunScriptName = "srunEuler.sh"

orthRunScript = r"""#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
export UCX_TLS=sm,self,ud_verbs
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
echo "UCX_TLS=${UCX_TLS}"
which mpirun
cp ../../../build/app/euler.exe .
ls -la euler.exe
mpirun euler.exe ./euler_config.json
rm euler.exe
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

print(json.dumps(baseConf, indent=4))



mults = [0.5,1,2,2.52,3.17,4,5.04,6.35,8]
# mults = [0.5,1,2,4,8]
# mults = [0.5,1,2,4,8, 0.25, 0.125, 0.0625]
# ODEs = [("HM3", 401,(0.55, 0, 1.333,0)), ("BDF2", 1,(0.55, 0, 1.333,0)), ("ESDIRK", 0,(0.55, 0, 1.333,0))]
# ODEs = [("BDF2", 1,(0.55, 0, 1.333,0))]
# ODEs = [("HM3L", 401, (0.5, 0, 1.2, 1))]
# ODEs = [("SDIRK", 101)]
# ODEs = [("ESDIRK", 0, (0.55, 0, 1.333,0))]
ODEs = []
# ODEs.append(("BDF2",   1,(0.55, 0, 1.333,0)))
# ODEs.append(("ESDIRK", 0,(0.55, 0, 1.333,0)))
# ODEs.append(("HM3",    401,(0.55, 0, 1.333,0)))
# ODEs.append(("HM3LOB", 401,(0.50, 0, 1, 0)))
# ODEs.append(("HM3U3R1", 401,(0.50, 0, 1.333, 2)))
ODEs.append(("HM3U2R1", 401,(0.25, 0, 1, 1)))

writtens = []
for iT in range(1,len(mults) + 1):
    for ODE in ODEs:
        caseName = casePrefix + "%d_%s" % (iT, ODE[0])
        mult = mults[iT - 1]
        confCur = copy.deepcopy(baseConf)
        confCur["timeMarchControl"]["dtImplicit"] = float(dtBase * mult)
        confCur["timeMarchControl"]["nTimeStep"] = int(round(stepBase / mult))
        confCur["timeMarchControl"]["odeCode"] = int(ODE[1])
        confCur["timeMarchControl"]["odeSetting1"] = ODE[2][0]
        confCur["timeMarchControl"]["odeSetting2"] = ODE[2][1]
        confCur["timeMarchControl"]["odeSetting3"] = ODE[2][2]
        confCur["timeMarchControl"]["odeSetting4"] = ODE[2][3]
        confCur["implicitReconstructionControl"]["zeroRecForSteps"] = int(round(zeroRecForStepsBase / mult))
        
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
   
        
        

