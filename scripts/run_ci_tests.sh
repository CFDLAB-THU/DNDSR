#!/usr/bin/env bash
# Run one cumulative test level. Build and install its targets before calling.
set -euo pipefail
cd "$(dirname "$0")/.."

level=${1:?usage: run_ci_tests.sh LEVEL BUILD_DIR PYTHON}
export DNDS_BUILD_DIR="$(realpath "${2:?missing build directory}")"
python_bin="$(realpath -s "${3:?missing Python interpreter}")"
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

test -x "$DNDS_BUILD_DIR/app/euler.exe"
"$python_bin" -S scripts/verify_ci_python.py
"$python_bin" -S scripts/verify_ci_python.py --h5py-first
mpiexec -n 2 "$python_bin" -S scripts/verify_ci_python.py --parallel

critical_runtime=(
    test/DNDS/test_runtime_array_bridge.py::test_one_pod_schema_spans_native_python_mpi_and_direct_io
    test/DNDS/test_runtime_transient.py::test_shared_context_known_route_and_tag_reuse
)

critical_mesh=(
    test/Geom/test_native_mesh_helpers.py::test_default_native_archive_file_faults_reject_collectively_and_retry
    test/CFV/test_fv_facility.py::test_reconstruction_shared_buffers_explicit_halo_and_point_sampling
)
pytest_args=(--with-mpi --timeout=600 --strict-markers -ra)
if [[ "$level" == 0 ]]; then
    "$python_bin" -S -m pytest "${pytest_args[@]}" \
        "${critical_runtime[@]}" "${critical_mesh[@]}" \
        test/DNDS/test_view_ownership.py::test_flat_view_retains_allocation_not_array \
        test/EulerP/test_basic_eulerP.py \
        test/Euler/test_restart_redistribute.py::test_restart_redistribute_same_np \
        test/Euler/test_restart_redistribute.py::test_restart_redistribute_different_np
    mpiexec -n 2 "$python_bin" -S -m pytest "${pytest_args[@]}" "${critical_runtime[@]}" "${critical_mesh[@]}" test/Cauchy/test_operators.py
else
    level_filter=()
    if [[ "$level" == 1 ]]; then
        level_filter=(-m 'not level2')
    fi
    "$python_bin" -S -m pytest "${pytest_args[@]}" "${level_filter[@]}" test/
    mpiexec -n 2 "$python_bin" -S -m pytest "${pytest_args[@]}" "${level_filter[@]}" \
        test/DNDS/test_runtime_*.py test/demos/test_runtime_solver.py \
        "${critical_mesh[@]}" test/Cauchy/test_operators.py
fi
