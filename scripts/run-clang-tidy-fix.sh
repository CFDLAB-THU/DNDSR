#!/usr/bin/env bash
# scripts/run-clang-tidy-fix.sh
#
# Thin shim: run the Python clang-tidy driver with --fix (uses the
# narrow .clang-tidy-fix profile and applies auto-fixes in place).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_clang_tidy.py" --fix "$@"
