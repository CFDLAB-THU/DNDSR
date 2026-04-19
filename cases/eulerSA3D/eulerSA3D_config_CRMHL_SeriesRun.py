import os, sys
import math
import copy

NNode = 20

JName = "test0-O2-NRW"


slurmJNamePre = "CRMHL_"

# alphas = [2.78, 7.05, 11.29, 17.05, 19.57, 20.55, 21.47]
alphas = [ 11.29]

exeSolver = "../build/app/eulerSA3D.exe"
caseConf = "../cases/eulerSA3D_config_CRMHL.json"
srunApp = "./srunApp.sh"

outBase = "../data/out/CRM/HLPW-4_CRM-HL_40-37_Nominal_v1a_Unstr-Hex-Prism-Pyr-Tet_Level-A_PW_V2_Q1"

kvs = {
    "/restartState/lastRestartFile": 
     "\\\"../data/out/CRM/HLPW-4_CRM-HL_40-37_Nominal_v1a_Unstr-Hex-Prism-Pyr-Tet_Level-A_PW_V2_Q1/test0-O2-Starter__C_p1280_restart.3000.dir\\\""
}

kvs["/vfvSettings/maxOrder"] = "1"
kvs["/vfvSettings/intOrder"] = "1"

cmds = []
for alpha in alphas:
    caseName = f"{JName}_{alpha}"
    slurmCaseName = slurmJNamePre + caseName
    kvs_c = copy.deepcopy(kvs)
    kvs_c["/dataIOControl/outPltName"] = f"\\\"{os.path.join(outBase, caseName)}\\\""
    ux = math.cos(alpha / 180.0 * math.pi)
    uz = math.sin(alpha / 180.0 * math.pi)
    kvs_c["/eulerSettings/farFieldStaticValue/1"] = str(ux)
    kvs_c["/eulerSettings/farFieldStaticValue/3"] = str(uz)
    kvs_c["/bcSettings/0/value/1"] = str(ux)
    kvs_c["/bcSettings/0/value/3"] = str(uz)
    
    cmd = f"sbatch -N{NNode} -J {slurmCaseName} -o {slurmCaseName}.txt {srunApp} {exeSolver} {caseConf} "
    for k,v in kvs_c.items():
        cmd += f"-k {k} -v {v} "
        
    cmds.append(cmd)

    
print("To Execute: ")
for i,cmd in enumerate(cmds) :
    print(f"=== No.{i+1}:")
    print(cmd)
print("say 'execute' to execute")
says = input()
if says == 'execute':
    for cmd in cmds:
        os.system(cmd)
        
    



