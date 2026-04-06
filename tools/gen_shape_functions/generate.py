#!/usr/bin/env python3
"""
Generate per-element C++ shape function headers.

Produces one .hpp file per element type in src/Geom/Elements/.
Each file contains template functions ShapeFunc_<ElemName>_Diff<0..3>
with CSE-optimized derivative code.

Usage:
    /usr/bin/python3 -m tools.gen_shape_functions.generate [--element NAME]

Run from the project root directory.
"""

import argparse
import os
import sys
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tools.gen_shape_functions.elements import ALL_ELEMENTS
from tools.gen_shape_functions.emitter import emit_element_file


def main():
    parser = argparse.ArgumentParser(description="Generate shape function C++ headers")
    parser.add_argument("--element", "-e", type=str, default=None,
                        help="Generate only the named element (e.g. 'Line2', 'Pyramid14')")
    parser.add_argument("--outdir", "-o", type=str,
                        default=os.path.join(PROJECT_ROOT, "src", "Geom", "Elements"),
                        help="Output directory for generated headers")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    elements = ALL_ELEMENTS
    if args.element:
        elements = [e for e in ALL_ELEMENTS if e.name == args.element]
        if not elements:
            names = [e.name for e in ALL_ELEMENTS]
            print(f"Error: unknown element '{args.element}'. Available: {names}",
                  file=sys.stderr)
            sys.exit(1)

    total_lines = 0
    for elem_cls in elements:
        out_path = os.path.join(args.outdir, f"{elem_cls.name}.hpp")
        print(f"Generating {elem_cls.name} -> {os.path.relpath(out_path, PROJECT_ROOT)}",
              file=sys.stderr)
        t0 = time.time()
        n_lines = emit_element_file(elem_cls, out_path)
        dt = time.time() - t0
        print(f"  -> {n_lines} lines in {dt:.1f}s", file=sys.stderr)
        total_lines += n_lines

    print(f"\nDone: {len(elements)} elements, {total_lines} total lines generated.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
