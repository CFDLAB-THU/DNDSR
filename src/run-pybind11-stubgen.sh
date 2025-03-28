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

export PYTHONPATH=$PYTHONPATH:$(pwd)
pybind11-stubgen DNDS -o pybind11-stubgen-out

# cp -v DNDS/_internal/dnds_pybind11/*.pyi DNDS/

for file in pybind11-stubgen-out/DNDS/_internal/dnds_pybind11/*.pyi; do
    cat "$file" >> "pybind11-stubgen-out/DNDS/$(basename "$file")"
    echo "$file -> pybind11-stubgen-out/DNDS/$(basename "$file")"
done

pybind11-stubgen Geom -o pybind11-stubgen-out

for file in pybind11-stubgen-out/Geom/_internal/geom_pybind11/*.pyi; do
    cat "$file" >> "pybind11-stubgen-out/Geom/$(basename "$file")"
    echo "$file -> pybind11-stubgen-out/Geom/$(basename "$file")"
done

cp -rv pybind11-stubgen-out/* .

rm -r pybind11-stubgen-out

cd $CURRENT_DIR




