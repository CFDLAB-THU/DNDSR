#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export LD_LIBRARY_PATH=$SCRIPT_DIR/../../build/install/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRIPT_DIR/../../external/Linux-x86_64/lib:$LD_LIBRARY_PATH