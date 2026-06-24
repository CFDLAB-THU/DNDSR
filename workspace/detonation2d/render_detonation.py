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


def compute_window_size(mesh, h_pad_factor: float = 1.04, annotation_factor: float = 2.0):
    """Return (W, H) so domain fills full width with proportional annotation space.

    Domain pixel height = W / (domain_aspect * h_pad_factor).
    H = domain_px * annotation_factor  (2.0 reproduces old 5000x1000 for 10:1).
    """
    domain_w = mesh.bounds[1] - mesh.bounds[0]
    domain_h = mesh.bounds[3] - mesh.bounds[2]
    domain_aspect = domain_w / domain_h if domain_h > 1e-30 else 10.0
    W = 5000
    domain_px = W / (domain_aspect * h_pad_factor)
    H = int(domain_px * annotation_factor)
    H = ((H + 7) // 8) * 8   # macroblock-aligned for video encoding
    H = max(400, min(H, 4000))
    return (W, H)


def render_frame(
    mesh: pv.UnstructuredGrid,
    field: str,
    clim: tuple,
    time_val: float,
    out_path: str,
    cmap: str = "plasma",
    window_size: tuple = None,
    h_pad_factor: float = 1.04,
    font_scale: float = 2.0,
):
    if window_size is None:
        window_size = compute_window_size(mesh, h_pad_factor)
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


def _filter_existing(data_dir, file_list):
    """Return only (fname, tval) pairs where fpath exists, with per-file warning."""
    existing, missing = [], 0
    for fname, tval in file_list:
        if os.path.exists(os.path.join(data_dir, fname)):
            existing.append((fname, tval))
        else:
            print(f"  WARN missing: {fname}")
            missing += 1
    if missing:
        print(f"  ({missing} files in series not on disk, skipped)")
    return existing


def cmd_render(args):
    # Single-frame mode: --series points to a .vtkhdf file
    if args.series.endswith(".vtkhdf"):
        fpath = args.series
        if not os.path.exists(fpath):
            sys.exit(f"File not found: {fpath}")
        data_dir = os.path.dirname(os.path.abspath(fpath))
        output_dir = os.path.join(data_dir, "pics")
        os.makedirs(output_dir, exist_ok=True)
        mesh = pv.read(fpath)
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        for field in args.field:
            clim = (args.clim_min, args.clim_max) if args.clim_min is not None and args.clim_max is not None else get_field_clim(
                mesh, field)
            out_path = os.path.join(output_dir, f"{base_name}_{field}.png")
            render_frame(mesh, field, clim, 0.0, out_path,
                         cmap=args.cmap,
                         window_size=tuple(
                             args.window) if args.window else None,
                         h_pad_factor=args.hpad, font_scale=args.font_scale)
            print(f"  {os.path.basename(out_path)}")
        print(f"Done. 1 frame x {len(args.field)} fields -> {output_dir}")
        return

    series_path = args.series
    data_dir = os.path.dirname(os.path.abspath(series_path))
    output_dir = os.path.join(data_dir, "pics")
    os.makedirs(output_dir, exist_ok=True)

    files = _filter_existing(data_dir, load_series(series_path))

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

    window = tuple(args.window) if args.window else None
    hpad = args.hpad
    font_scale = args.font_scale

    # Determine clim source
    last_found = selected[-1]
    clim_source_mesh = None
    clim_source_label = ""

    # Parse per-field --clim specs
    per_field_clims = {}
    for spec in args.clim:
        if "=" not in spec or ":" not in spec:
            sys.exit(f"Invalid --clim spec '{spec}': use FIELD=LOW:HIGH")
        field, range_str = spec.split("=", 1)
        low, high = range_str.split(":", 1)
        per_field_clims[field.strip()] = (
            float(low.strip()), float(high.strip()))

    if args.clim_min is not None and args.clim_max is not None:
        global_override = (args.clim_min, args.clim_max)
    else:
        global_override = None

    if args.clim_from:
        clim_from_fpath = os.path.join(data_dir, args.clim_from)
        if not os.path.exists(clim_from_fpath):
            for fname, _ in files:
                if extract_step(fname) == str(args.clim_from):
                    clim_from_fpath = os.path.join(data_dir, fname)
                    break
            else:
                sys.exit(f"Clim-from step/file not found: {args.clim_from}")
        clim_source_mesh = pv.read(clim_from_fpath)
        clim_source_label = args.clim_from

    global_clims = {}
    for field in args.field:
        if field in per_field_clims:
            clim = per_field_clims[field]
            print(
                f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}] (per-field)")
        elif global_override:
            clim = global_override
            print(
                f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}] (global manual)")
        elif clim_source_mesh:
            clim = get_field_clim(clim_source_mesh, field)
            print(
                f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}] (from {clim_source_label})")
        elif args.global_clim:
            clim = get_global_clim(data_dir, [f[0] for f in selected], field)
            print(f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}] (global)")
        else:
            last_fpath = os.path.join(data_dir, last_found[0])
            last_mesh = pv.read(last_fpath)
            clim = get_field_clim(last_mesh, field)
            print(
                f"  {field} clim: [{clim[0]:.4e}, {clim[1]:.4e}] (from last-found {last_found[0]})")
        global_clims[field] = clim

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


def cmd_all(args):
    """render + combine in one step."""
    cmd_render(args)
    cmd_combine(args)


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
        n_missing = 0
        for fname, _time in files:
            base = os.path.splitext(fname)[0]
            png_name = f"{base}_{field}.png"
            png_path = os.path.join(pics_dir, png_name)
            if os.path.exists(png_path):
                ordered.append(png_path)
            else:
                n_missing += 1
        if n_missing:
            print(
                f"  WARN {field}: {n_missing} PNGs missing of {len(files)} series entries")

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
                          help="Path to .vtkhdf.series file or single .vtkhdf file")
    p_render.add_argument("--field", action="append", default=[],
                          help="Scalar field(s) to render (repeatable)")
    p_render.add_argument("-i", "--index", default="all",
                          help="Series indices: N, start:stop:step, or 'all'")
    p_render.add_argument("--step", type=str, default=None,
                          help="Match step number in filename")
    p_render.add_argument("--stride", type=int, default=None,
                          help="Render every Nth frame")
    p_render.add_argument("--window", nargs=2, type=int,
                          default=None, help="Window width height (auto if unset)")
    p_render.add_argument("--hpad", type=float, default=1.04,
                          help="Horizontal padding factor")
    p_render.add_argument("--font-scale", type=float,
                          default=2.0, help="Font scale multiplier")
    p_render.add_argument("--clim-min", type=float,
                          default=None, help="Fixed colorbar minimum (applies to all fields)")
    p_render.add_argument("--clim-max", type=float,
                          default=None, help="Fixed colorbar maximum (applies to all fields)")
    p_render.add_argument("--clim", action="append", default=[],
                          help="Per-field range: FIELD=LOW:HIGH (e.g. T=300:4000 P=0:25)")
    p_render.add_argument("--clim-from", type=str, default=None,
                          help="Use this step number's data for colorbar range (e.g. '500' or 't_0.005000')")
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

    # ---- all (render + combine) ----
    p_all = sub.add_parser("all", help="Render frames then combine into video")
    p_all.add_argument("--series", required=True,
                       help="Path to .vtkhdf.series file")
    p_all.add_argument("--field", action="append", default=[],
                       help="Scalar field(s) to render and combine (repeatable)")
    p_all.add_argument("-i", "--index", default="all",
                       help="Series indices: N, start:stop:step, or 'all'")
    p_all.add_argument("--step", type=str, default=None,
                       help="Match step number in filename")
    p_all.add_argument("--stride", type=int, default=None,
                       help="Render every Nth frame")
    p_all.add_argument("--window", nargs=2, type=int,
                       default=None, help="Window width height (auto if unset)")
    p_all.add_argument("--hpad", type=float, default=1.04,
                       help="Horizontal padding factor")
    p_all.add_argument("--font-scale", type=float, default=2.0,
                       help="Font scale multiplier")
    p_all.add_argument("--clim-min", type=float, default=None,
                       help="Fixed colorbar minimum")
    p_all.add_argument("--clim-max", type=float, default=None,
                       help="Fixed colorbar maximum (applies to all fields)")
    p_all.add_argument("--clim", action="append", default=[],
                       help="Per-field range: FIELD=LOW:HIGH (e.g. T=300:4000 P=0:25)")
    p_all.add_argument("--clim-from", type=str, default=None,
                       help="Use this step number's data for colorbar range")
    p_all.add_argument("--cmap", type=str, default="plasma",
                       help="Colormap: plasma, inferno, viridis, turbo, hot, coolwarm, RdBu, magma, cividis, Blues, Reds, etc.")
    p_all.add_argument("--jobs", "-j", type=int, default=1,
                       help="Parallel workers (1 = serial, 0 = cpu_count)")
    p_all.add_argument("--global-clim", action="store_true",
                       help="Use global min/max across all selected frames")
    p_all.add_argument("--fps", type=int, default=24,
                       help="Video frames per second")
    p_all.add_argument("--crf", type=int, default=18,
                       help="Video quality (lower = better, 18-28)")

    args = parser.parse_args()

    if args.command == "render":
        if not args.field:
            p_render.error("At least one --field is required")
        cmd_render(args)
    elif args.command == "combine":
        cmd_combine(args)
    elif args.command == "all":
        if not args.field:
            p_all.error("At least one --field is required")
        cmd_all(args)


if __name__ == "__main__":
    main()
