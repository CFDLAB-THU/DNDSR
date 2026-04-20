#!/usr/bin/env bash
# serve_docs.sh — Build and serve documentation locally.
#
# Usage:
#   ./docs/serve_docs.sh                 # build Sphinx + serve on port 8000
#   ./docs/serve_docs.sh 9090            # build Sphinx + serve on port 9090
#   ./docs/serve_docs.sh --doxygen       # build Doxygen HTML + serve on port 8001
#   ./docs/serve_docs.sh --no-build      # skip build, serve existing Sphinx docs
#   ./docs/serve_docs.sh --no-build --doxygen  # skip build, serve existing Doxygen docs
#
# The script expects to be run from the project root directory, or it
# will attempt to detect the root from its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Defaults
MODE="sphinx"         # sphinx | doxygen
SKIP_BUILD=0
PORT=""

# --- Parse arguments ---
for arg in "$@"; do
    case "$arg" in
        --doxygen)   MODE="doxygen" ;;
        --no-build)  SKIP_BUILD=1 ;;
        [0-9]*)      PORT="$arg" ;;
    esac
done

if [[ "$MODE" == "doxygen" ]]; then
    DOC_HTML_DIR="${BUILD_DIR}/docs/html"
    BUILD_TARGET="doxygen"
    DEFAULT_PORT=8001
else
    DOC_HTML_DIR="${BUILD_DIR}/docs/sphinx"
    BUILD_TARGET="sphinx"
    DEFAULT_PORT=8000
fi

PORT="${PORT:-$DEFAULT_PORT}"

# --- Build docs if requested ---
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    if [[ ! -f "${BUILD_DIR}/Doxyfile" ]]; then
        echo "Error: ${BUILD_DIR}/Doxyfile not found."
        echo "Run CMake configure first:  cmake -B build"
        exit 1
    fi

    echo "Building ${MODE} docs..."
    cmake --build "$BUILD_DIR" -t "$BUILD_TARGET"
    echo ""
fi

# --- Verify HTML output exists ---
if [[ ! -f "${DOC_HTML_DIR}/index.html" ]]; then
    echo "Error: ${DOC_HTML_DIR}/index.html not found."
    echo "Build docs first:  cmake --build build -t ${BUILD_TARGET}"
    exit 1
fi

# --- Serve ---
echo "Serving ${MODE} docs at http://localhost:${PORT}"
echo "Press Ctrl-C to stop."
echo ""
cd "$DOC_HTML_DIR"
python3 -m http.server "$PORT"
