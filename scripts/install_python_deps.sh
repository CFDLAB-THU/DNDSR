#!/usr/bin/env bash
# install_python_deps.sh — Install ALL Python dependencies for DNDSR.
#
# This is the single entry point for setting up the Python environment.
# It installs:
#   1. Binary-wheel packages from requirements.txt  (numpy, scipy, pytest, …)
#   2. mpi4py compiled from source against the project's MPI
#   3. h5py   compiled from source against the project's HDF5 (cfd_externals)
#   4. scikit-build-core and pybind11 for building the C++ Python modules
#
# h5py and mpi4py MUST be compiled from source.  Binary wheels from PyPI
# bundle their own HDF5/MPI libraries, which conflict with the versions
# linked by the DNDSR pybind11 modules and cause crashes at import time.
#
# Usage:
#   ./scripts/install_python_deps.sh          # uses venv/bin/pip, auto-detect jobs
#   JOBS=8 ./scripts/install_python_deps.sh   # limit parallel compilation
#   PIP=path/to/pip ./scripts/install_python_deps.sh
#
# Environment variables:
#   PIP          — pip executable          (default: venv/bin/pip)
#   JOBS         — parallel make jobs      (default: $(nproc))
#   HDF5_DIR     — HDF5 install prefix     (default: external/cfd_externals/install)
#   MPI_CC       — MPI C compiler wrapper  (default: mpicc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIP="${PIP:-$PROJECT_ROOT/venv/bin/pip}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
HDF5_DIR="${HDF5_DIR:-$PROJECT_ROOT/external/cfd_externals/install}"
MPI_CC="${MPI_CC:-mpicc}"

if [ ! -x "$PIP" ]; then
    echo "Error: pip not found at $PIP" >&2
    echo "Create a venv first:  python3 -m venv venv && venv/bin/pip install --upgrade pip" >&2
    exit 1
fi

if [ ! -d "$HDF5_DIR/include" ]; then
    echo "Error: HDF5_DIR=$HDF5_DIR does not contain include/" >&2
    echo "Build external dependencies first:  cd external/cfd_externals && python cfd_externals_build.py" >&2
    exit 1
fi

echo "=== install_python_deps.sh ==="
echo "  PIP       = $PIP"
echo "  JOBS      = $JOBS"
echo "  HDF5_DIR  = $HDF5_DIR"
echo "  MPI_CC    = $MPI_CC"
echo ""

# ---- 1. Binary-wheel packages from requirements.txt ----------------------
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "--- Installing packages from requirements.txt ---"
    "$PIP" install --quiet -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found, skipping" >&2
fi

# ---- 2. Build tools needed for pybind11 modules --------------------------
echo ""
echo "--- Installing build tools (scikit-build-core, pybind11) ---"
"$PIP" install --quiet scikit-build-core pybind11

# ---- 3. mpi4py — compiled against the project's MPI ----------------------
echo ""
echo "--- Installing mpi4py (from source, CC=$MPI_CC) ---"
CC="$MPI_CC" \
    MAKEFLAGS="-j$JOBS" \
    "$PIP" install --no-binary mpi4py mpi4py --force-reinstall \
    2>&1 | tail -5

# ---- 4. h5py — compiled against the project's HDF5 (MPI-enabled) ---------
echo ""
echo "--- Installing h5py (from source, HDF5_DIR=$HDF5_DIR, HDF5_MPI=ON) ---"
CC="$MPI_CC" \
    HDF5_DIR="$HDF5_DIR" \
    HDF5_MPI="ON" \
    MAKEFLAGS="-j$JOBS" \
    "$PIP" install --no-binary h5py h5py --force-reinstall \
    2>&1 | tail -5

echo ""
echo "=== Done ==="
"$PIP" list | grep -iE "mpi4py|h5py|numpy|scipy|pytest"
