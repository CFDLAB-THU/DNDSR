#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. The optional argument names a new output directory.
checkout=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
data_root=$(dirname -- "$(git -C "$checkout" rev-parse --path-format=absolute --git-common-dir)")
mesh=${SUBOFF_MESH:-$data_root/data/mesh/SUBOFF_AFF8-01.cgns}
exe=${CAUCHY_EXE:-$checkout/build-acm-cuda/app/cauchy.exe}
config=$checkout/cases/cauchy/acm_suboff_p1_gg_amgx.json
if [[ ! -r $mesh || ! -x $exe ]]; then
    printf 'Need readable SUBOFF mesh and executable. Checked %s and %s\n' "$mesh" "$exe" >&2
    exit 2
fi
mesh=$(realpath -- "$mesh")
exe=$(realpath -- "$exe")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export DNDS_MT_OMP_NUM_THREADS=${DNDS_MT_OMP_NUM_THREADS:-$OMP_NUM_THREADS}
export DNDS_DIST_OMP_NUM_THREADS=${DNDS_DIST_OMP_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=1
output=${1:-$data_root/data/outUnsteady/Cauchy/SUBOFF_P1_AMGX/$(date +%Y%m%dT%H%M%S)}
output=$(realpath -m -- "$output")
json_string() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }
overrides=(
    -k /domains/0/settings/mesh/file -v "$(json_string "$mesh")"
    -k /domains/0/settings/execution/hostThreads -v "$DNDS_DIST_OMP_NUM_THREADS"
    -k /output/prefix -v "$(json_string "$output/run")"
)
if [[ ${SUBOFF_CHECK_CONFIG:-0} == 1 ]]; then
    exec "$exe" --check-config "$config" "${overrides[@]}"
fi
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
printf 'SUBOFF output directory %s\n' "$output"
mpirun --bind-to none -np 1 "$exe" "$config" "${overrides[@]}" 2>&1 | tee "$output/stdout.log"
