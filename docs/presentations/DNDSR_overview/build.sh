#!/usr/bin/env bash
#
# build.sh — assemble the DNDSR overview Marp deck from its parts.
#
# Source layout:
#   docs/presentations/DNDSR_overview/
#     00_frontmatter.md           YAML + <style> block (generated header comment)
#     parts/
#       00_title.md                Title slide
#       01_chapter_1.md            Chapter 1 (opening)
#       02_chapter_2.md            Chapter 2 (architecture)
#       ...
#       09_chapter_9.md            Chapter 9 (close)
#     res_manifest.txt             List of source → target image mappings
#                                  (populates docs/presentations/res/)
#     render_mermaid.py            Pre-renders ```mermaid blocks to SVG
#
# Output:
#   docs/presentations/res/                        (images, copied + Mermaid SVGs)
#   docs/presentations/DNDSR_overview.md           (always regenerated)
#   docs/presentations/DNDSR_overview.pdf          (if --pdf)
#   docs/presentations/DNDSR_overview.html         (if --html)
#
# Usage:
#   bash build.sh              # regenerate the combined .md only
#   bash build.sh --pdf        # regenerate + render PDF via marp-cli
#   bash build.sh --html       # regenerate + render HTML via marp-cli
#   bash build.sh --pdf --html # both
#   bash build.sh --no-res     # skip the image-copy step
#   bash build.sh --no-mermaid # skip the Mermaid pre-render step
#
# Requires:
#   bash, awk (for concatenation) — always present
#   python3   (for render_mermaid.py, unless --no-mermaid)
#   mmdc      (mermaid-cli, unless --no-mermaid)
#     npm install -g @mermaid-js/mermaid-cli
#   marp-cli  (only when --pdf or --html is passed)
#     npm install -g @marp-team/marp-cli
#
# Mermaid diagrams are pre-rendered to SVGs under res/ BEFORE Marp runs,
# so they display identically in HTML, PDF, and PPTX output (Marp does
# not render ```mermaid natively for PDF/image export).
#

set -euo pipefail

# ---------------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
PARTS_DIR="${SRC_DIR}/parts"
OUT_DIR="$(cd "${SRC_DIR}/.." && pwd)"                    # docs/presentations/
RES_DIR="${OUT_DIR}/res"
RES_MANIFEST="${SRC_DIR}/res_manifest.txt"
REPO_ROOT="$(cd "${OUT_DIR}/../.." && pwd)"               # repo root (= two levels up)
OUT_MD="${OUT_DIR}/DNDSR_overview.md"
OUT_PDF="${OUT_DIR}/DNDSR_overview.pdf"
OUT_HTML="${OUT_DIR}/DNDSR_overview.html"

# ---------------------------------------------------------------- args
DO_PDF=0
DO_HTML=0
DO_RES=1
DO_MERMAID=1
VERBOSE=0

for arg in "$@"; do
    case "$arg" in
        --pdf)        DO_PDF=1 ;;
        --html)       DO_HTML=1 ;;
        --no-res)     DO_RES=0 ;;
        --no-mermaid) DO_MERMAID=0 ;;
        --verbose|-v) VERBOSE=1 ;;
        --help|-h)
            sed -n '2,46p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log() { [[ "$VERBOSE" == "1" ]] && echo "  $*" >&2 || true; }

# ---------------------------------------------------------------- sanity
if [[ ! -f "${SRC_DIR}/00_frontmatter.md" ]]; then
    echo "ERROR: missing ${SRC_DIR}/00_frontmatter.md" >&2
    exit 1
fi
if [[ ! -d "${PARTS_DIR}" ]]; then
    echo "ERROR: missing ${PARTS_DIR}" >&2
    exit 1
fi

# Collect part files in sorted order.
mapfile -t PARTS < <(find "${PARTS_DIR}" -maxdepth 1 -name '*.md' -print | LC_ALL=C sort)
if [[ ${#PARTS[@]} -eq 0 ]]; then
    echo "ERROR: no *.md files under ${PARTS_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------- resources
# Populate docs/presentations/res/ from the canonical image locations in the
# repo (docs/elements/, docs/theory/, ...). The manifest is a plain text
# file in the source tree — one image per line, SOURCE <whitespace> TARGET,
# where SOURCE is repo-relative and TARGET is res/-relative.
#
# Lines starting with '#' and blank lines are ignored.
if [[ "$DO_RES" == "1" ]]; then
    if [[ ! -f "${RES_MANIFEST}" ]]; then
        echo "ERROR: missing res manifest ${RES_MANIFEST}" >&2
        exit 1
    fi

    mkdir -p "${RES_DIR}"
    copied=0
    skipped=0
    missing=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        # strip leading/trailing whitespace, skip comments/blank
        trimmed="${line#"${line%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        [[ -z "$trimmed" || "${trimmed:0:1}" == "#" ]] && continue

        # split on first run of whitespace
        src="${trimmed%%[[:space:]]*}"
        tgt="${trimmed#*[[:space:]]}"
        tgt="${tgt#"${tgt%%[![:space:]]*}"}"

        src_abs="${REPO_ROOT}/${src}"
        tgt_abs="${RES_DIR}/${tgt}"

        if [[ ! -f "${src_abs}" ]]; then
            echo "  MISSING source: ${src}" >&2
            missing=$((missing + 1))
            continue
        fi

        if [[ -f "${tgt_abs}" ]] && \
           cmp -s "${src_abs}" "${tgt_abs}"; then
            log "up-to-date: ${tgt}"
            skipped=$((skipped + 1))
            continue
        fi

        mkdir -p "$(dirname "${tgt_abs}")"
        cp "${src_abs}" "${tgt_abs}"
        log "copied:     ${src} -> res/${tgt}"
        copied=$((copied + 1))
    done < "${RES_MANIFEST}"

    echo "Resources:   ${copied} copied, ${skipped} up-to-date, ${missing} missing"
    if [[ "$missing" -gt 0 ]]; then
        echo "ERROR: ${missing} source image(s) missing — fix res_manifest.txt" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------- assemble
echo "Assembling ${#PARTS[@]} parts into ${OUT_MD}"

{
    # 1. Frontmatter + CSS (ends with </style>).
    cat "${SRC_DIR}/00_frontmatter.md"

    # 2. Each part. Every part is expected to begin with a slide
    #    (either "---\n<!-- _class: chapter -->" for chapter cards, or a
    #    plain content slide starting with its own "---").
    #    We ensure a blank line between parts.
    for part in "${PARTS[@]}"; do
        log "appending $(basename "$part")"
        printf '\n'
        cat "$part"
    done
} > "${OUT_MD}"

# ---------------------------------------------------------------- mermaid
# Pre-render fenced ```mermaid blocks into SVG files under res/, and
# rewrite the combined deck in-place to reference them via ![](res/*.svg).
# Marp does not render ```mermaid blocks for PDF/image output, so this
# step is what makes diagrams appear consistently everywhere.
#
# Chrome discovery: render_mermaid.py defaults to /usr/bin/google-chrome,
# but GitHub runners and other environments may have it elsewhere. If
# MERMAID_CHROME is set in the env we honour it; otherwise we probe
# common names on PATH and fall through to the default if none match.
if [[ "$DO_MERMAID" == "1" ]]; then
    if grep -q '^```mermaid' "${OUT_MD}"; then
        if ! command -v mmdc >/dev/null 2>&1; then
            echo "ERROR: mmdc (mermaid-cli) not found on PATH. Install with:" >&2
            echo "    npm install -g @mermaid-js/mermaid-cli" >&2
            echo "  Or pass --no-mermaid to skip diagram rendering." >&2
            exit 1
        fi
        if ! command -v python3 >/dev/null 2>&1; then
            echo "ERROR: python3 required by render_mermaid.py" >&2
            exit 1
        fi

        # Resolve Chrome path for Puppeteer (used by mmdc).
        chrome_path="${MERMAID_CHROME:-}"
        if [[ -z "$chrome_path" ]]; then
            for cand in google-chrome google-chrome-stable chromium chromium-browser; do
                if command -v "$cand" >/dev/null 2>&1; then
                    chrome_path="$(command -v "$cand")"
                    break
                fi
            done
        fi
        if [[ -z "$chrome_path" ]]; then
            echo "ERROR: no Chrome/Chromium found on PATH. Set MERMAID_CHROME or install one of:" >&2
            echo "  google-chrome  google-chrome-stable  chromium  chromium-browser" >&2
            exit 1
        fi
        log "using chrome: $chrome_path"

        echo "Rendering Mermaid diagrams → ${RES_DIR}/"
        python3 "${SRC_DIR}/render_mermaid.py" \
            --input  "${OUT_MD}" \
            --outdir "${RES_DIR}" \
            --prefix mermaid \
            --image-path-prefix res \
            --chrome "${chrome_path}"
    fi
fi

# ---------------------------------------------------------------- stats
SLIDES=$(awk 'BEGIN{c=0; in_code=0}
              /^```/ { in_code = !in_code; next }
              !in_code && /^---[[:space:]]*$/ { c++ }
              END { print c-1 }' "${OUT_MD}")      # minus 1 for YAML frontmatter close
BYTES=$(wc -c < "${OUT_MD}")
LINES=$(wc -l < "${OUT_MD}")

echo "  slides:    ${SLIDES}"
echo "  lines:     ${LINES}"
echo "  bytes:     ${BYTES}"

# ---------------------------------------------------------------- render
if [[ "$DO_PDF" == "1" || "$DO_HTML" == "1" ]]; then
    if ! command -v marp >/dev/null 2>&1; then
        echo "ERROR: marp-cli not found on PATH. Install with:" >&2
        echo "    npm install -g @marp-team/marp-cli" >&2
        exit 1
    fi

    if [[ "$DO_PDF" == "1" ]]; then
        echo "Rendering PDF → ${OUT_PDF}"
        marp --pdf --allow-local-files "${OUT_MD}" -o "${OUT_PDF}"
    fi
    if [[ "$DO_HTML" == "1" ]]; then
        echo "Rendering HTML → ${OUT_HTML}"
        marp --html --allow-local-files "${OUT_MD}" -o "${OUT_HTML}"
        python3 "${SRC_DIR}/add_toc_nav.py" "${OUT_HTML}"
        echo "  injected slide TOC sidebar"
    fi
fi

echo "Done."
