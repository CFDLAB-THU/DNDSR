#!/bin/bash
# Generate type stubs for the DNDSR Python package.
#
# Usage:
#   ./scripts/generate-stubs.sh
#
# Prerequisites:
#   - pip install pybind11-stubgen
#   - The DNDSR package must be importable (pip install -e . first)
#
# Output:
#   stubs/DNDSR/           — raw pybind11-stubgen output
#   python/DNDSR/**/*.pyi  — copied into the package for PEP 561

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUBS_DIR="${PROJECT_ROOT}/stubs"
PYTHON_DIR="${PROJECT_ROOT}/python"

# Clean previous output
rm -rf "${STUBS_DIR}"
mkdir -p "${STUBS_DIR}"

echo "=== Generating stubs for DNDSR.DNDS ==="
pybind11-stubgen DNDSR.DNDS -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.Geom ==="
pybind11-stubgen DNDSR.Geom -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.CFV ==="
pybind11-stubgen DNDSR.CFV -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.EulerP ==="
pybind11-stubgen DNDSR.EulerP -o "${STUBS_DIR}" || exit 1

# For each module, flatten the _ext/<binding>/*.pyi submodule stubs
# into the package-level directory so that IDEs see them.
for MODULE in DNDS Geom CFV EulerP; do
    # Find the binding subdirectory (e.g. _ext/dnds_pybind11/)
    EXT_DIR=$(find "${STUBS_DIR}/DNDSR/${MODULE}/_ext" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true)
    if [ -n "${EXT_DIR}" ]; then
        for stub in "${EXT_DIR}"/*.pyi; do
            [ -f "$stub" ] || continue
            base=$(basename "$stub")
            dest="${STUBS_DIR}/DNDSR/${MODULE}/${base}"
            # Overwrite (not append) to avoid duplication
            cp -v "$stub" "$dest"
        done
    fi
done

# Copy stubs into the python package tree for PEP 561 compliance
echo "=== Copying stubs into python/DNDSR/ ==="
for MODULE in DNDS Geom CFV EulerP; do
    SRC="${STUBS_DIR}/DNDSR/${MODULE}"
    DST="${PYTHON_DIR}/DNDSR/${MODULE}"
    if [ -d "${SRC}" ]; then
        # Copy only .pyi files (not __pycache__ etc.)
        find "${SRC}" -name '*.pyi' | while read -r pyi; do
            rel="${pyi#${SRC}/}"
            mkdir -p "$(dirname "${DST}/${rel}")"
            cp -v "$pyi" "${DST}/${rel}"
        done
    fi
done

# Also copy the top-level DNDSR/__init__.pyi if generated
if [ -f "${STUBS_DIR}/DNDSR/__init__.pyi" ]; then
    cp -v "${STUBS_DIR}/DNDSR/__init__.pyi" "${PYTHON_DIR}/DNDSR/__init__.pyi"
fi

echo "=== Done. Stubs in ${STUBS_DIR}/ and ${PYTHON_DIR}/DNDSR/ ==="
