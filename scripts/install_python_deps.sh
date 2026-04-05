#!/usr/bin/env bash
# install_python_deps.sh — Install Python dependencies that need source builds.
#
# mpi4py and h5py must be compiled from source against the project's MPI
# and HDF5 libraries.  Binary wheels from PyPI link the wrong libraries
# and fail at import time.
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

# Standard test/runtime deps (binary wheels are fine)
echo "--- Installing binary deps ---"
"$PIP" install --quiet numpy scipy pytest pytest-mpi pytest-subtests

# mpi4py — must be compiled against the project's MPI
echo ""
echo "--- Installing mpi4py (from source, CC=$MPI_CC) ---"
CC="$MPI_CC" \
    MAKEFLAGS="-j$JOBS" \
    "$PIP" install --no-binary mpi4py mpi4py --force-reinstall \
    2>&1 | tail -5

# h5py — must be compiled against the project's HDF5 (which is MPI-enabled)
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
