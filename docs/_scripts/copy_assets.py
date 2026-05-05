#!/usr/bin/env python3
"""Copy documentation image/PDF assets into a build output tree.

Used by both the Doxygen and Sphinx docs pipelines so the same copy
logic (preserving subdirectory structure) produces collision-free
output for each generator.

The script walks one or more SOURCE roots and copies every file with a
supported extension into a DEST tree, keeping the relative directory
layout intact:

    docs/theory/PyramidShow.png   ->   <DEST>/theory/PyramidShow.png
    docs/elements/Tri3_nodes.png  ->   <DEST>/elements/Tri3_nodes.png

The copy is idempotent: files already present with identical content are
skipped. Missing DEST directories are created on demand.

Usage:

    python3 copy_assets.py --dest <DEST_DIR> \
        [--root <SRC_ROOT> ...] \
        [--rel-to <REL_BASE>] \
        [--ext .png .pdf .jpg .svg] \
        [--verbose]

If no --root is given, the current working directory is used (matching
the legacy `getAllAttachForDox.py` behavior).

`--rel-to` controls what part of each source path is preserved in the
output: if set, it is the directory the relative layout is computed
against; otherwise, paths are made relative to the source root that
contained them.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path


DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif")


def copy_tree(
    roots: list[Path],
    dest: Path,
    rel_to: Path | None,
    exts: tuple[str, ...],
    verbose: bool,
) -> tuple[int, int]:
    """Copy files matching *exts* from *roots* into *dest*.

    Returns (copied, up_to_date).
    """
    copied = 0
    up_to_date = 0
    exts_lower = tuple(e.lower() for e in exts)

    dest.mkdir(parents=True, exist_ok=True)

    for root in roots:
        root = root.resolve()
        if not root.exists():
            print(
                f"warning: source root does not exist: {root}", file=sys.stderr)
            continue

        base = rel_to.resolve() if rel_to is not None else root
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts_lower:
                continue

            try:
                rel = path.resolve().relative_to(base)
            except ValueError:
                # File outside --rel-to; fall back to relative from root.
                rel = path.resolve().relative_to(root)

            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and filecmp.cmp(str(path), str(target), shallow=False):
                up_to_date += 1
                continue

            shutil.copy2(str(path), str(target))
            copied += 1
            if verbose:
                print(f"  {rel}")

    return copied, up_to_date


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Copy docs assets into a build output directory."
    )
    ap.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory (created if missing).",
    )
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="Source root to scan (may be given multiple times). "
        "Defaults to current working directory.",
    )
    ap.add_argument(
        "--rel-to",
        type=Path,
        default=None,
        help="Directory that source-paths are expressed relative to in "
        "the destination. Defaults to each --root.",
    )
    ap.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help=f"File extensions to include (default: {' '.join(DEFAULT_EXTS)}).",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    roots = args.root or [Path.cwd()]
    copied, up_to_date = copy_tree(
        roots=roots,
        dest=args.dest,
        rel_to=args.rel_to,
        exts=tuple(args.ext),
        verbose=args.verbose,
    )

    total = copied + up_to_date
    print(
        f"copy_assets: {copied} copied, {up_to_date} up-to-date "
        f"({total} total) -> {args.dest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
