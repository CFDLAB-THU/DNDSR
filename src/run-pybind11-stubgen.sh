#!/bin/bash


CURRENT_DIR="$(pwd)"

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd $SCRIPT_DIR

# echo $(pwd)
# echo LD_LIBRARY_PATH
# printenv LD_LIBRARY_PATH
# echo PATH
# printenv PATH
# echo PYTHONPATH
# printenv PYTHONPATH
# ldd ../build_py/src/DNDS/libdnds_shared.so
# 
# ldd pybind11-stubgen

export PYTHONPATH=$PYTHONPATH:$(pwd)/..

### DNDS
pybind11-stubgen DNDSR.DNDS -o pybind11-stubgen-out || exit 1

# cp -v DNDS/_internal/dnds_pybind11/*.pyi DNDS/

for file in pybind11-stubgen-out/DNDSR/DNDS/_internal/dnds_pybind11/*.pyi; do
    dest=pybind11-stubgen-out/DNDSR/DNDS/$(basename "$file")
    cat "$file" >> $dest
    echo "$file -> $dest"
done

### Geom

pybind11-stubgen DNDSR.Geom -o pybind11-stubgen-out || exit 1

for file in pybind11-stubgen-out/DNDSR/Geom/_internal/geom_pybind11/*.pyi; do
    dest=pybind11-stubgen-out/DNDSR/Geom/$(basename "$file")
    cat "$file" >> $dest
    echo "$file -> $dest"
done

### CFV

pybind11-stubgen DNDSR.CFV -o pybind11-stubgen-out || exit 1

# cat pybind11-stubgen-out/CFV

for file in pybind11-stubgen-out/DNDSR/CFV/_internal/cfv_pybind11/*.pyi; do
    dest=pybind11-stubgen-out/DNDSR/CFV/$(basename "$file")
    cat "$file" >> $dest
    echo "$file -> $dest"
done

### copy

cp -rv pybind11-stubgen-out/DNDSR/* .

rm -r pybind11-stubgen-out

cd $CURRENT_DIR




