#!/usr/bin/env bash
# Regenerate JSON Schema files for all solver variants.
#
# Usage:
#   cd <project_root>
#   bash cases/update_schemas.sh [build_dir]
#
# The build directory defaults to "build".  All 6 solver executables
# must already be compiled.

set -euo pipefail

BUILD_DIR="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VARIANTS=(euler euler3D eulerSA eulerSA3D euler2EQ euler2EQ3D)

for v in "${VARIANTS[@]}"; do
    exe="${BUILD_DIR}/app/${v}.exe"
    out="${SCRIPT_DIR}/${v}_schema.json"
    if [[ ! -x "$exe" ]]; then
        echo "SKIP $v: $exe not found or not executable" >&2
        continue
    fi
    mpirun -np 1 "$exe" --emit-schema 2>/dev/null \
        | grep -v '^JSON:' > "$out"
    echo "  $v -> $(basename "$out")"
done

echo "Done."
