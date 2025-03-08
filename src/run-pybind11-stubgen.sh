#!/bin/bash


CURRENT_DIR="$(pwd)"

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd $SCRIPT_DIR

pybind11-stubgen DNDS -o .

# cp -v DNDS/_internal/dnds_pybind11/*.pyi DNDS/

for file in DNDS/_internal/dnds_pybind11/*.pyi; do
    cat "$file" >> "DNDS/$(basename "$file")"
    echo "$file -> DNDS/$(basename "$file")"
done

cd $CURRENT_DIR




