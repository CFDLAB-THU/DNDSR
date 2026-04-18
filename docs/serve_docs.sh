#!/usr/bin/env bash
# serve_docs.sh — Build Doxygen HTML docs and serve them locally.
#
# Usage:
#   ./docs/serve_docs.sh            # build + serve on port 8000
#   ./docs/serve_docs.sh 9090       # build + serve on port 9090
#   ./docs/serve_docs.sh --no-build # skip build, just serve existing docs
#
# The script expects to be run from the project root directory, or it
# will attempt to detect the root from its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
DOC_HTML_DIR="${BUILD_DIR}/docs/html"

PORT="${1:-8000}"
SKIP_BUILD=0

if [[ "${1:-}" == "--no-build" ]]; then
    SKIP_BUILD=1
    PORT="${2:-8000}"
elif [[ "${2:-}" == "--no-build" ]]; then
    SKIP_BUILD=1
fi

# --- Build docs if requested ---
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    if [[ ! -f "${BUILD_DIR}/Doxyfile" ]]; then
        echo "Error: ${BUILD_DIR}/Doxyfile not found."
        echo "Run CMake configure first:  cmake -B build"
        exit 1
    fi

    echo "Building docs..."
    cmake --build "$BUILD_DIR" -t docs
    echo ""
fi

# --- Verify HTML output exists ---
if [[ ! -f "${DOC_HTML_DIR}/index.html" ]]; then
    echo "Error: ${DOC_HTML_DIR}/index.html not found."
    echo "Build docs first:  cmake --build build -t docs"
    exit 1
fi

# --- Serve ---
echo "Serving docs at http://localhost:${PORT}"
echo "Press Ctrl-C to stop."
echo ""
cd "$DOC_HTML_DIR"
python3 -m http.server "$PORT"
