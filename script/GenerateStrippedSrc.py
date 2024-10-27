import os
import shutil
srcDir = "./src"
srcNewRoot = "./newSrc"

# use new: mv src srcOrig; ln -sf newSrc/src src
# use old: rm src; mv srcOrig src

def addPragmaOnceInFile(file:str):
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    f = open(file,"w")
    lines.insert(0, "#pragma once\n")
    f.writelines(lines)
    f.close()

for root, dirs, files in os.walk(srcDir):
    rootA = os.path.join(srcNewRoot, root)
    for filename in files:
        ret = 0
        os.makedirs(rootA, exist_ok=True)

        if filename.endswith((".hpp", ".h", ".cpp", ".hxx")):
            ret = os.system(f"gcc -fpreprocessed -E  -P -dD {os.path.join(root,filename)} -o {os.path.join(rootA, filename)}")
            if(filename.endswith((".hpp", ".h", ".hxx"))):
                addPragmaOnceInFile(os.path.join(rootA, filename))
        else:
            shutil.copy(os.path.join(root, filename), os.path.join(rootA, filename))

        if ret != 0:
            raise Warning("return bad")
