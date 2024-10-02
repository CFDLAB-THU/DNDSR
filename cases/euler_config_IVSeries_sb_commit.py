

import os

baseConf = "../cases/euler_config_IV.json"
exePath = "../build/app/euler.exe"
outPathPrefix = "../data/outUnsteady/IV10_PP"

NNode = 1

# IBis = [0,1,2,3,4]
IBis = [2]
N0 = 20
N0_Mesh = 10

# prefix = "IVPP_"
# suffix = "-sm"
# dt0 = 100e-4

prefix = "IVPP_"
suffix = "-sml"
dt0 = 0.5

for i in IBis:
    Nc = int(N0 * 2**i)
    Nc_Mesh = int(N0_Mesh * 2**i)
    dt = float(dt0 /(2 **i))
    caseName = f"{prefix}{Nc}{suffix}"
    outName = f"{outPathPrefix}/{caseName}"
    meshFile = f"../data/mesh/IV10_{Nc_Mesh}.cgns"
    cmd = f"./ssb.sh {NNode} {caseName} {exePath} {baseConf} "
    cmd = cmd + f"-k /dataIOControl/meshFile -v \\\"{meshFile}\\\" "
    cmd = cmd + f"-k /timeMarchControl/dtImplicit -v {dt} "
    cmd = cmd + f"-k /dataIOControl/meshDirectBisect -v {1} "
    cmd = cmd + f"-k /dataIOControl/outPltName -v \\\"{outName}\\\" "
    
    print("cmd is:\n" + cmd)
    
    ret = os.system(cmd)
    
    if ret:
        raise RuntimeError(f"return is {ret}")