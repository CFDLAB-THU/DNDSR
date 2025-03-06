#!/bin/bash


CURRENT_DIR="$(pwd)"

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd $SCRIPT_DIR

pybind11-stubgen DNDS -o .

cp -v DNDS/_internal/dnds_pybind11/*.pyi DNDS/

cd $CURRENT_DIR




