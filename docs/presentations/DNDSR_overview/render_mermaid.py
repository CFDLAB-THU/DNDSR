#!/usr/bin/env python3
"""Pre-render fenced ```mermaid blocks to SVG and rewrite the deck.

Scans *INPUT* (the assembled Marp deck), extracts each ```mermaid
code block, invokes `mmdc` (mermaid-cli) to render it to SVG, writes
the SVG into *OUTDIR*, and replaces the fenced block in-place with a
standard Markdown image reference:

    ![](res/mermaid_NN.svg)

This lets the deck render Mermaid diagrams consistently in Marp's
HTML, PDF, and PPTX outputs (Marp does not render ```mermaid natively
for PDF/image export).

Idempotency: the script hashes each block's source and skips mmdc when
the cached SVG already matches the hash (stored alongside the SVG).

Usage:

    python3 render_mermaid.py --input deck.md --outdir res/ --prefix mermaid

The deck is rewritten in place. Run before invoking Marp.

Configuration:

  - MMDC executable: `mmdc` on PATH (override with --mmdc).
  - Puppeteer executable: defaults to /usr/bin/google-chrome.
    Override with --chrome (empty string to use mmdc's default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


MERMAID_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```mermaid[ \t]*\n(?P<body>.*?)^\1```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def render_one(
    mmdc: str,
    source: str,
    out_svg: Path,
    chrome: str | None,
    theme: str,
    background: str,
) -> None:
    """Run `mmdc` once on *source* and write SVG to *out_svg*."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.mmd"
        in_path.write_text(source, encoding="utf-8")

        cmd = [
            mmdc,
            "-i", str(in_path),
            "-o", str(out_svg),
            "-t", theme,
            "-b", background,
        ]
        if chrome:
            cfg_path = Path(td) / "puppeteer.json"
            cfg_path.write_text(
                json.dumps({"executablePath": chrome,
                           "args": ["--no-sandbox"]}),
                encoding="utf-8",
            )
            cmd += ["-p", str(cfg_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"mmdc failed for {out_svg.name}:\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}\n"
                f"--- SOURCE ---\n{source}\n"
            )
            raise SystemExit(f"mmdc failed (exit {result.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-render Mermaid blocks in a Marp deck to SVG."
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="Assembled Markdown deck (rewritten in place).")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Directory where SVGs are written.")
    ap.add_argument("--prefix", default="mermaid",
                    help="Filename prefix for generated SVGs.")
    ap.add_argument("--mmdc", default="mmdc", help="Path to mmdc.")
    ap.add_argument("--chrome", default="/usr/bin/google-chrome",
                    help="Puppeteer executablePath. Empty to use default.")
    ap.add_argument("--theme", default="default",
                    choices=["default", "dark", "neutral", "forest"])
    ap.add_argument("--background", default="transparent")
    ap.add_argument("--image-path-prefix", default="res",
                    help="Path prefix used in the rewritten Markdown image "
                    "reference (relative to the deck).")
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    blocks = list(MERMAID_FENCE_RE.finditer(text))
    if not blocks:
        print("render_mermaid: no ```mermaid blocks found; nothing to do.")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    chrome = args.chrome if args.chrome else None

    # Build rewritten text piece by piece so indices stay correct.
    out_parts: list[str] = []
    last_end = 0
    n_rendered = 0
    n_cached = 0

    for idx, m in enumerate(blocks, start=1):
        body = m.group("body")
        # Stable 8-char hash of the block body so cached SVGs remain valid
        # across re-assembly when a diagram is unchanged.
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]
        svg_name = f"{args.prefix}_{idx:02d}_{h}.svg"
        svg_path = args.outdir / svg_name

        if not svg_path.exists():
            render_one(
                mmdc=args.mmdc,
                source=body,
                out_svg=svg_path,
                chrome=chrome,
                theme=args.theme,
                background=args.background,
            )
            n_rendered += 1
            print(f"  rendered  {svg_name}")
        else:
            n_cached += 1

        image_ref = f"{args.image_path_prefix}/{svg_name}"
        replacement = f"{m.group('indent')}![]({image_ref})\n"

        out_parts.append(text[last_end: m.start()])
        out_parts.append(replacement)
        last_end = m.end()

    out_parts.append(text[last_end:])
    new_text = "".join(out_parts)

    if new_text != text:
        args.input.write_text(new_text, encoding="utf-8")

    # Clean stale SVGs in outdir that no longer match any block hash.
    current_names = {
        f"{args.prefix}_{i:02d}_"
        f"{hashlib.sha256(m.group('body').encode()).hexdigest()[:8]}.svg"
        for i, m in enumerate(blocks, start=1)
    }
    stale = 0
    for existing in args.outdir.glob(f"{args.prefix}_*.svg"):
        if existing.name not in current_names:
            existing.unlink()
            stale += 1

    total = len(blocks)
    print(
        f"render_mermaid: {total} blocks "
        f"({n_rendered} rendered, {n_cached} cached, {stale} stale removed) "
        f"-> {args.outdir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
