#!/usr/bin/env bash
# Run one cumulative test level. Build and install its targets before calling.
set -euo pipefail
cd "$(dirname "$0")/.."

level=${1:?usage: run_ci_tests.sh LEVEL BUILD_DIR PYTHON}
export DNDS_BUILD_DIR="$(realpath "${2:?missing build directory}")"
python_bin="$(realpath -s "${3:?missing Python interpreter}")"
cache="$DNDS_BUILD_DIR/CMakeCache.txt"
mpi_launcher="$(sed -n 's/^MPIEXEC_EXECUTABLE:FILEPATH=//p' "$cache")"
test -x "$mpi_launcher"
# The restart fixture launches mpirun itself. Match both the launcher and
# mpi4py's dynamically loaded MPI to the build, including in a service shell.
export PATH="$(dirname "$mpi_launcher"):$PATH"
mpi_library="$(sed -n 's/^MPI_mpi_LIBRARY:FILEPATH=//p' "$cache")"
if [[ -f "$mpi_library" ]]; then
    export LD_LIBRARY_PATH="$(dirname "$mpi_library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export DNDS_MECH_PATH="${DNDS_MECH_PATH:-$PWD/external/cfd_externals/install/data}"
export CANTERA_DATA="$DNDS_MECH_PATH"
# The fixture writer must load the CGNS ABI selected by this build.
cgns_library="$(sed -n 's/^DNDS_EXTERNAL_LIB_CGNS:FILEPATH=//p' "$DNDS_BUILD_DIR/CMakeCache.txt")"
if [[ -n "$cgns_library" ]]; then
    export DNDS_EXTERNAL_LIB_CGNS="$cgns_library"
fi

case "$level" in
    0) labels='^level0$' ;;
    1) labels='^level[01]$' ;;
    2) labels='^level[012]$' ;;
    *) printf 'Invalid CI test level: %s\n' "$level" >&2; exit 2 ;;
esac

ctest --test-dir "$DNDS_BUILD_DIR" -L "$labels" -LE '^python$' \
    --no-tests=error --output-on-failure --timeout 600

# Bindings and h5py share one installed copy irrespective of import order.
# -S avoids editable startup hooks in reused development environments.
python_site="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$PWD/python:$DNDS_BUILD_DIR/test/cpp:$python_site"
export LD_LIBRARY_PATH="$PWD/python/DNDSR/_lib/dndsr_external${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1

if ((level >= 1)); then
    test -x "$DNDS_BUILD_DIR/app/euler.exe"
fi
"$python_bin" -S scripts/verify_ci_python.py
"$python_bin" -S scripts/verify_ci_python.py --h5py-first
"$mpi_launcher" -n 2 "$python_bin" -S scripts/verify_ci_python.py --parallel

pytest_args=(--with-mpi --timeout=600 --strict-markers -ra --dnds-test-level="$level")
"$python_bin" -S -m pytest "${pytest_args[@]}" --dnds-ci-lane=serial test/
rank_wrapper=()
if grep -Eiq '^DNDS_USE_CUDA:BOOL=(ON|1|TRUE|YES|Y)$' "$cache"; then
    rank_wrapper=("$python_bin" test/demos/one_rank_per_gpu.py)
fi
"$mpi_launcher" -n 2 "${rank_wrapper[@]}" "$python_bin" -S -m pytest "${pytest_args[@]}" --dnds-ci-lane=mpi test/
