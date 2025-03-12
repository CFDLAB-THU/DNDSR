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
pybind11-stubgen DNDS -o .

# cp -v DNDS/_internal/dnds_pybind11/*.pyi DNDS/

for file in DNDS/_internal/dnds_pybind11/*.pyi; do
    cat "$file" >> "DNDS/$(basename "$file")"
    echo "$file -> DNDS/$(basename "$file")"
done

cd $CURRENT_DIR




