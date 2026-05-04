#!/usr/bin/env bash
# scripts/run-clang-format.sh
#
# Thin backward-compatibility shim. The real implementation lives in
# scripts/run_clang_format.py -- see its --help for the full option list.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_clang_format.py" "$@"
