#!/usr/bin/env python3
"""Generate an N×N single-zone QUAD_4 CGNS mesh for periodic wave tests.

Creates a uniform N×N structured/unstructured mesh on [0,L] × [0,L] with
N² quad cells, (N+1)² nodes, and boundary face elements (BAR_2 edges) with
ZoneBC records, suitable for DNDSR periodic read with
translations [L,0,0] and [0,L,0].

Default: N=8, L=8.0, output data/mesh/Uniform_8x8_wave.cgns.
Also supports N=4, N=2 for coarse-grid exact operators.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


CG_MODE_WRITE = 1
Unstructured = 3
QUAD_4 = 7
BAR_2 = 3
RealDouble = 4
BCTypeNull = 0
PointList = 2
PointRange = 4
EdgeCenter = 8

cgsize_t = ctypes.c_long


def _load_cgns(repo_root: Path) -> ctypes.CDLL:
    lib = repo_root / "external" / "cfd_externals" / "install" / "lib" / "libcgns.so"
    if not lib.exists():
        raise FileNotFoundError(f"CGNS shared library not found: {lib}")
    cgns = ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    cgns.cg_get_error.restype = ctypes.c_char_p
    cgns.cg_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_close.argtypes = [ctypes.c_int]
    cgns.cg_base_write.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_zone_write.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(cgsize_t),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_coord_write.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_section_write.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        cgsize_t,
        cgsize_t,
        ctypes.c_int,
        ctypes.POINTER(cgsize_t),
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_boco_write.argtypes = [
        ctypes.c_int,          # fn
        ctypes.c_int,          # B
        ctypes.c_int,          # Z
        ctypes.c_char_p,       # boconame
        ctypes.c_int,          # bocotype
        ctypes.c_int,          # ptset_type
        cgsize_t,              # npnts
        ctypes.POINTER(cgsize_t),  # pnts
        ctypes.POINTER(ctypes.c_int),  # bc_id
    ]
    cgns.cg_boco_gridlocation_write.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return cgns


def _check(status: int, msg: str, cgns: ctypes.CDLL | None = None) -> None:
    if status != 0:
        detail = ""
        if cgns is not None:
            err = cgns.cg_get_error()
            if err:
                detail = f": {err.decode(errors='replace')}"
        raise RuntimeError(f"CGNS call failed: {msg}{detail}")


def _as_cgsize(values: list[int]):
    return (cgsize_t * len(values))(*values)


def _as_double(values: list[float]):
    return (ctypes.c_double * len(values))(*values)


def generate(out_path: Path, repo_root: Path, N: int = 8, L: float = 8.0,
             origin: float = 0.0) -> Path:
    """Generate the N×N single-zone QUAD_4 mesh and return the output path.

    Node coordinates are ``origin + i * h`` so the physical domain is
    ``[origin, origin + L] × [origin, origin + L]``.
    """
    cgns = _load_cgns(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    nnx, nny = N + 1, N + 1  # nodes per direction
    h = L / N                  # cell spacing

    # Node index helper (1-indexed): node(i,j) = j*nnx + i + 1
    def node_idx(i: int, j: int) -> int:
        return j * nnx + i + 1

    # Build node coordinates (row-major: i inner, j outer)
    coord_x: list[float] = []
    coord_y: list[float] = []
    for j in range(nny):
        for i in range(nnx):
            coord_x.append(origin + float(i) * h)
            coord_y.append(origin + float(j) * h)

    # Build cell connectivity (1-indexed, CCW from bottom-left)
    # Quad node ordering: BL, BR, TR, TL
    quad_connectivity: list[int] = []
    for j in range(N):
        for i in range(N):
            n0 = node_idx(i, j)
            n1 = node_idx(i + 1, j)
            n2 = node_idx(i + 1, j + 1)
            n3 = node_idx(i, j + 1)
            quad_connectivity.extend([n0, n1, n2, n3])

    n_nodes = len(coord_x)
    n_cells = N * N
    n_bnd_verts = 0

    # Build PERIODIC_1 boundary edges (left=MAIN, right=DONOR, 16 edges total)
    per1_main_edges: list[int] = []   # left boundary, x=0
    per1_donor_edges: list[int] = []  # right boundary, x=N
    for j in range(N):
        per1_main_edges.extend([node_idx(0, j), node_idx(0, j + 1)])
        per1_donor_edges.extend([node_idx(N, j), node_idx(N, j + 1)])

    # Build PERIODIC_2 boundary edges (bottom=MAIN, top=DONOR, 16 edges total)
    per2_main_edges: list[int] = []   # bottom boundary, y=0
    per2_donor_edges: list[int] = []  # top boundary, y=N
    for i in range(N):
        per2_main_edges.extend([node_idx(i, 0), node_idx(i + 1, 0)])
        per2_donor_edges.extend([node_idx(i, N), node_idx(i + 1, N)])

    fn = ctypes.c_int()
    _check(
        cgns.cg_open(str(out_path).encode(), CG_MODE_WRITE, ctypes.byref(fn)),
        "cg_open",
        cgns,
    )
    try:
        base = ctypes.c_int()
        _check(
            cgns.cg_base_write(fn.value, b"Base", 2, 2, ctypes.byref(base)),
            "cg_base_write",
            cgns,
        )

        zone = ctypes.c_int()
        sizes = _as_cgsize([n_nodes, n_cells, n_bnd_verts])
        _check(
            cgns.cg_zone_write(
                fn.value, base.value, b"Zone", sizes, Unstructured, ctypes.byref(
                    zone)
            ),
            "cg_zone_write",
            cgns,
        )

        coord_id = ctypes.c_int()
        _check(
            cgns.cg_coord_write(
                fn.value, base.value, zone.value, RealDouble,
                b"CoordinateX", _as_double(coord_x), ctypes.byref(coord_id),
            ),
            "CoordinateX",
            cgns,
        )
        _check(
            cgns.cg_coord_write(
                fn.value, base.value, zone.value, RealDouble,
                b"CoordinateY", _as_double(coord_y), ctypes.byref(coord_id),
            ),
            "CoordinateY",
            cgns,
        )

        # --- Volume section: QUAD_4, elements 1..64 ---
        section = ctypes.c_int()
        _check(
            cgns.cg_section_write(
                fn.value, base.value, zone.value, b"QuadElements",
                QUAD_4, 1, n_cells, 0, _as_cgsize(quad_connectivity),
                ctypes.byref(section),
            ),
            "cg_section_write QuadElements",
            cgns,
        )

        # --- Boundary section: PERIODIC_1 (MAIN, left edges, elements n_cells+1..n_cells+N) ---
        e1_start = n_cells + 1
        e1_end = n_cells + N
        _check(
            cgns.cg_section_write(
                fn.value, base.value, zone.value, b"PERIODIC_1",
                BAR_2, e1_start, e1_end, 0, _as_cgsize(per1_main_edges),
                ctypes.byref(section),
            ),
            "cg_section_write PERIODIC_1",
            cgns,
        )

        # --- Boundary section: PERIODIC_1_DONOR (right edges, elements n_cells+N+1..n_cells+2N) ---
        e1d_start = e1_end + 1
        e1d_end = e1_end + N
        _check(
            cgns.cg_section_write(
                fn.value, base.value, zone.value, b"PERIODIC_1_DONOR",
                BAR_2, e1d_start, e1d_end, 0, _as_cgsize(per1_donor_edges),
                ctypes.byref(section),
            ),
            "cg_section_write PERIODIC_1_DONOR",
            cgns,
        )

        # --- Boundary section: PERIODIC_2 (MAIN, bottom edges, elements n_cells+2N+1..n_cells+3N) ---
        e2_start = e1d_end + 1
        e2_end = e1d_end + N
        _check(
            cgns.cg_section_write(
                fn.value, base.value, zone.value, b"PERIODIC_2",
                BAR_2, e2_start, e2_end, 0, _as_cgsize(per2_main_edges),
                ctypes.byref(section),
            ),
            "cg_section_write PERIODIC_2",
            cgns,
        )

        # --- Boundary section: PERIODIC_2_DONOR (top edges, elements n_cells+3N+1..n_cells+4N) ---
        e2d_start = e2_end + 1
        e2d_end = e2_end + N
        _check(
            cgns.cg_section_write(
                fn.value, base.value, zone.value, b"PERIODIC_2_DONOR",
                BAR_2, e2d_start, e2d_end, 0, _as_cgsize(per2_donor_edges),
                ctypes.byref(section),
            ),
            "cg_section_write PERIODIC_2_DONOR",
            cgns,
        )

        # --- ZoneBC: PERIODIC_1 (MAIN, elements 65..72) ---
        bc_id = ctypes.c_int()
        p1_range = _as_cgsize([e1_start, e1_end])
        _check(
            cgns.cg_boco_write(
                fn.value, base.value, zone.value, b"PERIODIC_1",
                BCTypeNull, PointRange, 2, p1_range, ctypes.byref(bc_id),
            ),
            "cg_boco_write PERIODIC_1",
            cgns,
        )
        _check(
            cgns.cg_boco_gridlocation_write(
                fn.value, base.value, zone.value, bc_id.value, EdgeCenter,
            ),
            "cg_boco_gridlocation_write PERIODIC_1",
            cgns,
        )

        # --- ZoneBC: PERIODIC_1_DONOR (elements 73..80) ---
        p1d_range = _as_cgsize([e1d_start, e1d_end])
        _check(
            cgns.cg_boco_write(
                fn.value, base.value, zone.value, b"PERIODIC_1_DONOR",
                BCTypeNull, PointRange, 2, p1d_range, ctypes.byref(bc_id),
            ),
            "cg_boco_write PERIODIC_1_DONOR",
            cgns,
        )
        _check(
            cgns.cg_boco_gridlocation_write(
                fn.value, base.value, zone.value, bc_id.value, EdgeCenter,
            ),
            "cg_boco_gridlocation_write PERIODIC_1_DONOR",
            cgns,
        )

        # --- ZoneBC: PERIODIC_2 (MAIN, elements 81..88) ---
        p2_range = _as_cgsize([e2_start, e2_end])
        _check(
            cgns.cg_boco_write(
                fn.value, base.value, zone.value, b"PERIODIC_2",
                BCTypeNull, PointRange, 2, p2_range, ctypes.byref(bc_id),
            ),
            "cg_boco_write PERIODIC_2",
            cgns,
        )
        _check(
            cgns.cg_boco_gridlocation_write(
                fn.value, base.value, zone.value, bc_id.value, EdgeCenter,
            ),
            "cg_boco_gridlocation_write PERIODIC_2",
            cgns,
        )

        # --- ZoneBC: PERIODIC_2_DONOR (elements 89..96) ---
        p2d_range = _as_cgsize([e2d_start, e2d_end])
        _check(
            cgns.cg_boco_write(
                fn.value, base.value, zone.value, b"PERIODIC_2_DONOR",
                BCTypeNull, PointRange, 2, p2d_range, ctypes.byref(bc_id),
            ),
            "cg_boco_write PERIODIC_2_DONOR",
            cgns,
        )
        _check(
            cgns.cg_boco_gridlocation_write(
                fn.value, base.value, zone.value, bc_id.value, EdgeCenter,
            ),
            "cg_boco_gridlocation_write PERIODIC_2_DONOR",
            cgns,
        )
    finally:
        _check(cgns.cg_close(fn.value), "cg_close", cgns)

    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate an NxN single-zone QUAD_4 CGNS mesh."
    )
    parser.add_argument(
        "--N", type=int, default=8,
        help="Number of cells per direction (default: 8)",
    )
    parser.add_argument(
        "--L", type=float, default=8.0,
        help="Domain size [0,L] × [0,L] (default: 8.0)",
    )
    parser.add_argument(
        "--origin", type=float, default=0.0,
        help="Domain origin shift (default: 0.0). Node coords = origin + i*L/N.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output path (default: data/mesh/Uniform_{N}x{N}_wave.cgns)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = repo_root / "data" / "mesh" / \
            f"Uniform_{args.N}x{args.N}_wave.cgns"

    lib_dir = repo_root / "external" / "cfd_externals" / "install" / "lib"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    path = generate(args.output, repo_root, N=args.N,
                    L=args.L, origin=args.origin)
    print(f"Generated: {path}")


if __name__ == "__main__":
    main()
