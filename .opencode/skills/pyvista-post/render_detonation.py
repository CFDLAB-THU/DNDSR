#!/usr/bin/env python3
"""Render 2D detonation VTKHDF output frames and combine into video.

Usage:
  # Render frames
  python render_detonation.py render --series SERIES_FILE --field T --field P

  # Single test frame
  python render_detonation.py render --series SERIES_FILE --step 500 --field T

  # Combine PNGs into video (one video per field)
  python render_detonation.py combine --series SERIES_FILE

  # Combine with custom fps
  python render_detonation.py combine --series SERIES_FILE --fps 30
"""

import argparse
import json
import os
import re as _re
import subprocess
import sys

import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_step(filename: str):
    m = _re.search(r"__(\d+|t_[\d.]+)\.vtkhdf$", filename)
    return m.group(1) if m else filename


def parse_index_spec(spec: str, n_total: int):
    if spec.lower() == "all":
        return list(range(n_total))
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else n_total
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return [i for i in range(start, min(stop, n_total), step)]
    idx = int(spec)
    if 0 <= idx < n_total:
        return [idx]
    return []


def load_series(series_path: str):
    with open(series_path) as f:
        series = json.load(f)
    return sorted(
        [(e["name"], e["time"]) for e in series["files"]],
        key=lambda x: x[1],
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_frame(
    mesh: pv.UnstructuredGrid,
    field: str,
    clim: tuple,
    time_val: float,
    out_path: str,
    cmap: str = "plasma",
    window_size: tuple = (5000, 1000),
    h_pad_factor: float = 1.04,
    font_scale: float = 2.0,
):
    W, H = window_size
    domain_w = mesh.bounds[1] - mesh.bounds[0]
    cx = (mesh.bounds[0] + mesh.bounds[1]) / 2
    cy = (mesh.bounds[2] + mesh.bounds[3]) / 2
    aspect = W / H

    ps = domain_w * h_pad_factor / (2 * aspect)

    base_fs = 24 * font_scale
    title_fs = 28 * font_scale
    text_fs = 22 * font_scale

    pl = pv.Plotter(off_screen=True, window_size=[W, H])
    pl.background_color = "white"

    pl.add_mesh(
        mesh,
        scalars=field,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            "title": f"{field}",
            "vertical": False,
            "position_x": 0.2,
            "position_y": 0.08,
            "width": 0.6,
            "height": 0.06,
            "label_font_size": int(base_fs),
            "title_font_size": int(title_fs),
            "fmt": "%.2g",
        },
    )

    pl.camera.parallel_projection = True
    pl.camera.position = (cx, cy, 1.0)
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.view_up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = ps

    pl.add_text(f"t = {time_val:.4e}", position="upper_edge",
                font_size=int(text_fs), color="black")

    pl.screenshot(out_path)
    pl.close()


def get_field_clim(mesh, field: str):
    data = mesh[field]
    if data.ndim == 2:
        data = np.linalg.norm(data, axis=1)
    return (float(np.nanmin(data)), float(np.nanmax(data)))


def get_global_clim(data_dir: str, files: list, field: str, sample_count: int = 10):
    sample_files = files[:: max(1, len(files) // sample_count)]
    global_min = float("inf")
    global_max = float("-inf")
    for fname in sample_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        m = pv.read(fpath)
        data = m[field]
        if data.ndim == 2:
            data = np.linalg.norm(data, axis=1)
        global_min = min(global_min, float(np.nanmin(data)))
        global_max = max(global_max, float(np.nanmax(data)))
    return (global_min, global_max)


def _render_one_frame(args_tuple):
    """Worker for multiprocessing: renders all fields for a single VTKHDF file."""
    (
        fname, data_dir, fields, clims, time_val,
        cmap, window_size, hpad, font_scale,
    ) = args_tuple

    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return f"SKIP missing: {fpath}"

    try:
        base_name = os.path.splitext(fname)[0]
        mesh = pv.read(fpath)
        results = []
        for field in fields:
            out_name = f"{base_name}_{field}.png"
            out_path = os.path.join(data_dir, "pics", out_name)
            render_frame(mesh, field, clims[field], time_val, out_path,
                         cmap=cmap, window_size=window_size,
                         h_pad_factor=hpad, font_scale=font_scale)
            results.append(out_name)
        return f"OK {fname} -> {', '.join(results)}"
    except Exception as e:
        return f"FAIL {fname}: {e}"


def cmd_render(args):
    series_path = args.series
    data_dir = os.path.dirname(os.path.abspath(series_path))
    output_dir = os.path.join(data_dir, "pics")
    os.makedirs(output_dir, exist_ok=True)

    files = load_series(series_path)

    if args.step:
        step_target = str(args.step)
        indices = [i for i, (fname, _) in enumerate(
            files) if extract_step(fname) == step_target]
        if not indices:
            sys.exit(f"No frame matching step '{args.step}'")
    else:
        indices = parse_index_spec(args.index, len(files))
    if args.stride:
        indices = indices[:: args.stride]

    selected = [files[i] for i in indices if i < len(files)]
    if not selected:
        sys.exit("No frames selected")

    window = tuple(args.window)
    hpad = args.hpad
    font_scale = args.font_scale

    global_clims = {}
    for field in args.field:
        if args.clim_min is not None and args.clim_max is not None:
            clim = (args.clim_min, args.clim_max)
        elif args.global_clim:
            clim = get_global_clim(data_dir, [f[0] for f in selected], field)
        else:
            last_fpath = os.path.join(data_dir, selected[-1][0])
            last_mesh = pv.read(last_fpath)
            clim = get_field_clim(last_mesh, field)
        global_clims[field] = clim
        print(f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}]")

    # Build task list — one task per VTKHDF file (all fields rendered from same mesh)
    tasks = []
    for fname, time_val in selected:
        tasks.append((
            fname, data_dir, args.field, global_clims, time_val,
            args.cmap, window, hpad, font_scale,
        ))

    n_jobs = args.jobs
    if n_jobs == 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Rendering {len(tasks)} tasks with {n_jobs} workers ...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_render_one_frame, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                print(f"  [{i}/{len(tasks)}] {result}")
    else:
        for t in tasks:
            result = _render_one_frame(t)
            print(f"  {result}")

    print(
        f"Done. {len(selected)} frames x {len(args.field)} fields -> {output_dir}")


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def cmd_combine(args):
    import tempfile

    series_path = args.series
    data_dir = os.path.dirname(os.path.abspath(series_path))
    pics_dir = os.path.join(data_dir, "pics")

    if not os.path.isdir(pics_dir):
        sys.exit(f"pics directory not found: {pics_dir}")

    files = load_series(series_path)

    png_files_all = sorted(
        [f for f in os.listdir(pics_dir) if f.endswith(".png")])
    if args.field:
        fields = args.field
    else:
        fields = sorted(set(
            f.rsplit("_", 1)[-1].replace(".png", "")
            for f in png_files_all
        ))
    print(f"Fields: {fields}")

    series_base = os.path.basename(series_path).replace(".vtkhdf.series", "")

    for field in fields:
        ordered = []
        for fname, _time in files:
            base = os.path.splitext(fname)[0]
            png_name = f"{base}_{field}.png"
            png_path = os.path.join(pics_dir, png_name)
            if os.path.exists(png_path):
                ordered.append(png_path)

        if not ordered:
            print(f"  No PNGs for field {field}, skipping")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, src in enumerate(ordered):
                dst = os.path.join(tmpdir, f"{i:06d}.png")
                os.symlink(os.path.abspath(src), dst)

            out_video = os.path.join(pics_dir, f"{series_base}{field}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", os.path.join(tmpdir, "%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(args.crf),
                out_video,
            ]
            print(f"  Encoding {field}: {len(ordered)} frames -> {out_video}")
            subprocess.run(cmd, check=True)

    print(f"Done. Videos in {pics_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D detonation VTKHDF render and video tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- render ----
    p_render = sub.add_parser("render", help="Render VTKHDF frames to PNG")
    p_render.add_argument("--series", required=True,
                          help="Path to .vtkhdf.series file")
    p_render.add_argument("--field", action="append", default=[],
                          help="Scalar field(s) to render (repeatable)")
    p_render.add_argument("-i", "--index", default="all",
                          help="Series indices: N, start:stop:step, or 'all'")
    p_render.add_argument("--step", type=str, default=None,
                          help="Match step number in filename")
    p_render.add_argument("--stride", type=int, default=None,
                          help="Render every Nth frame")
    p_render.add_argument("--window", nargs=2, type=int,
                          default=[5000, 1000], help="Window width height")
    p_render.add_argument("--hpad", type=float, default=1.04,
                          help="Horizontal padding factor")
    p_render.add_argument("--font-scale", type=float,
                          default=2.0, help="Font scale multiplier")
    p_render.add_argument("--clim-min", type=float,
                          default=None, help="Fixed colorbar minimum")
    p_render.add_argument("--clim-max", type=float,
                          default=None, help="Fixed colorbar maximum")
    p_render.add_argument("--cmap", type=str, default="plasma",
                          help="Colormap: plasma, inferno, viridis, turbo, hot, coolwarm, RdBu, magma, cividis, Blues, Reds, etc.")
    p_render.add_argument("--jobs", "-j", type=int, default=1,
                          help="Parallel workers (1 = serial, 0 = cpu_count)")
    p_render.add_argument("--global-clim", action="store_true",
                          help="Use global min/max across all selected frames")

    # ---- combine ----
    p_combine = sub.add_parser(
        "combine", help="Combine rendered PNGs into video")
    p_combine.add_argument("--series", required=True,
                           help="Path to .vtkhdf.series file")
    p_combine.add_argument("--field", action="append", default=[],
                           help="Field(s) to combine (repeatable; default: auto-detect all)")
    p_combine.add_argument("--fps", type=int, default=24,
                           help="Frames per second")
    p_combine.add_argument("--crf", type=int, default=18,
                           help="Video quality (lower = better, 18-28)")

    args = parser.parse_args()

    if args.command == "render":
        if not args.field:
            p_render.error("At least one --field is required")
        cmd_render(args)
    elif args.command == "combine":
        cmd_combine(args)


if __name__ == "__main__":
    main()
