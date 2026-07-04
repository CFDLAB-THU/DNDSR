#!/usr/bin/env python3
"""Generate small one-sided multiblock CGNS meshes for DNDSR tests.

The generated files are intended for the ignored `data/mesh/` tree (or the
external `cfd_meshes` CI repository).  Each block is an unstructured zone with a
single QUAD_4 cell and duplicated interface vertices.  Zone connectivity is
written only in the right/up direction, so the files exercise one-sided
Abutting1to1 deduplication.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


CG_MODE_WRITE = 1
Vertex = 2
Abutting1to1 = 4
PointList = 2
PointListDonor = 3
Integer = 2
RealDouble = 4
Unstructured = 3
QUAD_4 = 7


cgsize_t = ctypes.c_long


def _load_cgns(repo_root: Path) -> ctypes.CDLL:
    lib = repo_root / "external" / "cfd_externals" / "install" / "lib" / "libcgns.so"
    if not lib.exists():
        raise FileNotFoundError(f"CGNS shared library not found: {lib}")
    cgns = ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    cgns.cg_get_error.restype = ctypes.c_char_p
    cgns.cg_open.argtypes = [ctypes.c_char_p,
                             ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_close.argtypes = [ctypes.c_int]
    cgns.cg_base_write.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                   ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_zone_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(
        cgsize_t), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_coord_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    cgns.cg_section_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                      ctypes.c_int, cgsize_t, cgsize_t, ctypes.c_int, ctypes.POINTER(cgsize_t), ctypes.POINTER(ctypes.c_int)]
    cgns.cg_conn_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, cgsize_t, ctypes.POINTER(
        cgsize_t), ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, cgsize_t, ctypes.POINTER(cgsize_t), ctypes.POINTER(ctypes.c_int)]
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


def _write_conn(cgns, fn: int, base: int, zone: int, name: str, donor: str,
                points: list[int], donor_points: list[int]) -> None:
    conn_idx = ctypes.c_int()
    _check(
        cgns.cg_conn_write(
            fn,
            base,
            zone,
            name.encode(),
            Vertex,
            Abutting1to1,
            PointList,
            len(points),
            _as_cgsize(points),
            donor.encode(),
            Unstructured,
            PointListDonor,
            Integer,
            len(donor_points),
            _as_cgsize(donor_points),
            ctypes.byref(conn_idx),
        ),
        f"cg_conn_write {name}",
        cgns,
    )


def generate(blocks: int, out_dir: Path, repo_root: Path) -> Path:
    cgns = _load_cgns(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"GeneratedMultiBlock{blocks}x{blocks}_OneSided.cgns"
    if path.exists():
        path.unlink()

    fn = ctypes.c_int()
    _check(cgns.cg_open(str(path).encode(), CG_MODE_WRITE,
           ctypes.byref(fn)), "cg_open", cgns)
    try:
        base = ctypes.c_int()
        _check(cgns.cg_base_write(fn.value, b"Base", 2, 2,
               ctypes.byref(base)), "cg_base_write", cgns)

        zone_ids: dict[tuple[int, int], int] = {}
        for by in range(blocks):
            for bx in range(blocks):
                zone = ctypes.c_int()
                # 4 local vertices, 1 quad cell, 0 boundary vertices.
                sizes = _as_cgsize([4, 1, 0])
                zname = f"B{bx}_{by}"
                _check(
                    cgns.cg_zone_write(fn.value, base.value, zname.encode(
                    ), sizes, Unstructured, ctypes.byref(zone)),
                    f"cg_zone_write {zname}",
                    cgns,
                )
                zone_ids[(bx, by)] = zone.value

                xs = [float(bx), float(bx + 1), float(bx + 1), float(bx)]
                ys = [float(by), float(by), float(by + 1), float(by + 1)]
                coord = ctypes.c_int()
                _check(cgns.cg_coord_write(fn.value, base.value, zone.value, RealDouble,
                       b"CoordinateX", _as_double(xs), ctypes.byref(coord)), "CoordinateX", cgns)
                _check(cgns.cg_coord_write(fn.value, base.value, zone.value, RealDouble,
                       b"CoordinateY", _as_double(ys), ctypes.byref(coord)), "CoordinateY", cgns)

                section = ctypes.c_int()
                # Local node order: 1 BL, 2 BR, 3 TR, 4 TL.
                quad = _as_cgsize([1, 2, 3, 4])
                _check(cgns.cg_section_write(fn.value, base.value, zone.value, b"Elem",
                       QUAD_4, 1, 1, 0, quad, ctypes.byref(section)), "cg_section_write", cgns)

        # One-sided connectivity: write only right and upper interfaces.
        for by in range(blocks):
            for bx in range(blocks):
                zid = zone_ids[(bx, by)]
                zname = f"B{bx}_{by}"
                if bx + 1 < blocks:
                    donor = f"B{bx + 1}_{by}"
                    _write_conn(cgns, fn.value, base.value, zid,
                                f"{zname}_right", donor, [2, 3], [1, 4])
                if by + 1 < blocks:
                    donor = f"B{bx}_{by + 1}"
                    _write_conn(cgns, fn.value, base.value, zid,
                                f"{zname}_top", donor, [4, 3], [1, 2])
    finally:
        _check(cgns.cg_close(fn.value), "cg_close", cgns)
    return path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=repo_root / "data" / "mesh")
    parser.add_argument("--blocks", type=int, nargs="*",
                        default=[2, 3], choices=[2, 3])
    args = parser.parse_args()

    lib_dir = repo_root / "external" / "cfd_externals" / "install" / "lib"
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for n in args.blocks:
        print(generate(n, args.output, repo_root))


if __name__ == "__main__":
    main()
