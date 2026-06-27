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


def compute_window_size(mesh_or_region, h_pad_factor: float = 1.04,
                        fill_frac: float = 0.75, width: int = 5000,
                        margin: float = 0.03, top_strip: float = 0.12,
                        cb_strip: float = 0.12):
    """Return (W, H) so region fills width and `fill_frac` of data viewport height.

    Accounts for viewport layout: data viewport occupies
    (1 - 2*margin) × (1 - top_strip - cb_strip) fraction of canvas.
    """
    if isinstance(mesh_or_region, tuple):
        rxmin, rxmax, rymin, rymax = mesh_or_region
    else:
        rxmin, rxmax = mesh_or_region.bounds[0], mesh_or_region.bounds[1]
        rymin, rymax = mesh_or_region.bounds[2], mesh_or_region.bounds[3]
    region_w = rxmax - rxmin
    region_h = rymax - rymin
    region_aspect = region_w / region_h if region_h > 1e-30 else 10.0
    data_w_frac = 1 - 2 * margin
    data_h_frac = 1 - top_strip - cb_strip
    viewport_ratio = data_w_frac / (data_h_frac * fill_frac)
    W = width
    domain_px = W / (region_aspect * h_pad_factor)
    H = int(domain_px * viewport_ratio)
    H = ((H + 7) // 8) * 8
    H = max(400, min(H, 2 * W))
    return (W, H)


def render_frame(
    mesh: pv.UnstructuredGrid,
    field: str,
    clim: tuple,
    time_val: float,
    out_path: str,
    cmap: str = "plasma",
    render_width: int = 5000,
    render_dpi: int = 144,
    h_pad_factor: float = 1.04,
    font_scale: float = 2.0,
    font_scale_time: float = 1.0,
    font_scale_cb: float = 1.0,
    font_scale_axes: float = 1.0,
    margin: float = 0.03,
    top_strip: float = 0.12,
    cb_strip: float = 0.12,
    render_region: tuple = None,
):
    window_size = compute_window_size(
        render_region if render_region else mesh, h_pad_factor,
        width=render_width, margin=margin,
        top_strip=top_strip, cb_strip=cb_strip)
    W, H = window_size

    # Render region in world coords
    if render_region:
        rxmin, rxmax, rymin, rymax = render_region
    else:
        rxmin, rxmax = mesh.bounds[0], mesh.bounds[1]
        rymin, rymax = mesh.bounds[2], mesh.bounds[3]
    region_w = rxmax - rxmin
    region_h = rymax - rymin
    cx = (rxmin + rxmax) / 2
    cy = (rymin + rymax) / 2

    # Viewport maths — strips have horizontal margin, bg renderer fills gaps
    data_vp = (margin, cb_strip, 1 - margin, 1 - top_strip)
    data_w = data_vp[2] - data_vp[0]
    data_h = data_vp[3] - data_vp[1]
    data_aspect = W * data_w / (H * data_h)

    ps = region_w * h_pad_factor / (2 * data_aspect)

    base_fs = 24 * font_scale
    title_fs = 28 * font_scale
    text_fs = 22 * font_scale

    # Layered multi-renderer: bg + three strips, no black gaps
    pl = pv.Plotter(off_screen=True, window_size=[W, H],
                    shape=(4, 1), border=False)

    # --- Renderer 0: full-canvas white background ---
    pl.subplot(0, 0)
    pl.background_color = "white"
    pl.camera.parallel_projection = True

    # --- Renderer 1: top strip (time label) ---
    pl.subplot(1, 0)
    pl.background_color = "white"
    ta = pl.add_text(f"t = {time_val:.4e}", position="lower_edge",
                     font_size=int(0.5 * text_fs * font_scale_time), color="black")
    ta.prop.font_family = "times"
    pl.camera.parallel_projection = True

    # --- Renderer 2: data (mesh + axes + time label in data viewport) ---
    pl.subplot(2, 0)
    pl.background_color = "white"
    actor = pl.add_mesh(
        mesh,
        scalars=field,
        cmap=cmap,
        clim=clim,
        show_edges=False,
        show_scalar_bar=False,
    )
    pl.camera.parallel_projection = True
    pl.camera.position = (cx, cy, 1.0)
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.view_up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = ps
    ca = pl.show_bounds(
        location="all", ticks="both",
        xtitle="x", ytitle="y",
        font_size=int(0.125 * base_fs * font_scale_axes), color="black",
        n_xlabels=5, n_ylabels=3,
        bounds=(rxmin, rxmax, rymin, rymax, 0, 0),
    )
    ca.SetScreenSize(10.0 * 72.0 / render_dpi)
    ca.x_label_format = "%.4g"
    ca.y_label_format = "%.4g"
    ca.label_offset = 2.0 * font_scale_axes
    ca.title_offset = (2.0 * font_scale_axes, 5.0 * font_scale_axes)
    for prop_name in ("x_axis_label_text_property", "y_axis_label_text_property",
                      "x_axis_title_text_property", "y_axis_title_text_property"):
        if hasattr(ca, prop_name):
            getattr(ca, prop_name).font_family = "times"

    # --- Renderer 3: colorbar strip ---
    pl.subplot(3, 0)
    pl.background_color = "white"
    pl.add_scalar_bar(
        title=f"{field}",
        mapper=actor.mapper,
        vertical=False,
        position_x=0.20,
        position_y=0.45,
        width=0.60,
        height=0.30,
        label_font_size=int(base_fs * font_scale_cb),
        title_font_size=int(title_fs * font_scale_cb),
        color="black",
        font_family="times",
        fmt="%.2g",
    )

    # Viewports: bg full-canvas, strips inset with margins
    pl.renderers[0].SetViewport(0, 0, 1, 1)
    pl.renderers[1].SetViewport(margin, 1 - top_strip, 1 - margin, 1)
    pl.renderers[2].SetViewport(
        data_vp[0], data_vp[1], data_vp[2], data_vp[3])
    pl.renderers[3].SetViewport(margin, 0, 1 - margin, cb_strip)

    if hasattr(pl, 'render_window'):
        pl.render_window.SetDPI(render_dpi)

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


def _render_one_frame(kwargs):
    """Worker for multiprocessing: renders all fields for a single VTKHDF file."""
    fname = kwargs["fname"]
    data_dir = kwargs["data_dir"]
    fields = kwargs["fields"]
    clims = kwargs["clims"]
    time_val = kwargs["time_val"]
    cmap = kwargs["cmap"]
    render_width = kwargs["render_width"]
    render_dpi = kwargs["render_dpi"]
    font_scale = kwargs["font_scale"]
    font_scale_time = kwargs.get("font_scale_time", 1.0)
    font_scale_cb = kwargs.get("font_scale_cb", 1.0)
    font_scale_axes = kwargs.get("font_scale_axes", 1.0)
    hpad = kwargs.get("hpad", 1.04)
    margin = kwargs.get("margin", 0.03)
    top_strip = kwargs.get("top_strip", 0.12)
    cb_strip = kwargs.get("cb_strip", 0.12)
    render_region = kwargs.get("render_region")

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
                         cmap=cmap, render_width=render_width,
                         render_dpi=render_dpi,
                         font_scale=font_scale,
                         font_scale_time=font_scale_time,
                         font_scale_cb=font_scale_cb,
                         font_scale_axes=font_scale_axes,
                         h_pad_factor=hpad,
                         margin=margin, top_strip=top_strip,
                         cb_strip=cb_strip, render_region=render_region)
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
                         render_width=args.render_width,
                         render_dpi=args.render_dpi,
                         font_scale=args.font_scale,
                         font_scale_time=args.font_scale_time,
                         font_scale_cb=args.font_scale_cb,
                         font_scale_axes=args.font_scale_axes,
                         h_pad_factor=args.hpad,
                         margin=args.margin, top_strip=args.top_strip,
                         cb_strip=args.cb_strip)
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

    render_width = args.render_width
    render_dpi = args.render_dpi
    font_scale = args.font_scale
    font_scale_time = args.font_scale_time
    font_scale_cb = args.font_scale_cb
    font_scale_axes = args.font_scale_axes
    hpad = args.hpad
    margin = args.margin
    top_strip = args.top_strip
    cb_strip = args.cb_strip

    # Parse render region
    render_region = None
    if args.render_x_lim or args.render_y_lim:
        rx = [float(x) for x in args.render_x_lim.split(
            ":")] if args.render_x_lim else None
        ry = [float(x) for x in args.render_y_lim.split(
            ":")] if args.render_y_lim else None
        render_region = (
            rx[0] if rx else 0.0, rx[1] if rx else float("inf"),
            ry[0] if ry else 0.0, ry[1] if ry else float("inf"),
        )

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
        tasks.append(dict(
            fname=fname, data_dir=data_dir, fields=args.field,
            clims=global_clims, time_val=time_val,
            cmap=args.cmap, render_width=render_width,
            render_dpi=render_dpi,
            font_scale=font_scale, font_scale_time=font_scale_time,
            font_scale_cb=font_scale_cb, font_scale_axes=font_scale_axes,
            hpad=hpad, margin=margin, top_strip=top_strip,
            cb_strip=cb_strip, render_region=render_region,
        ))

    n_jobs = args.jobs
    if n_jobs == 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1:
        import signal
        from concurrent.futures import ProcessPoolExecutor, as_completed

        pool = None

        def _on_signal(signum, frame):
            if pool:
                print(f"\n  Caught signal {signum}, shutting down workers ...")
                pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

        old_sigint = signal.signal(signal.SIGINT, _on_signal)
        old_sigterm = signal.signal(signal.SIGTERM, _on_signal)

        try:
            print(f"Rendering {len(tasks)} tasks with {n_jobs} workers ...")
            pool = ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=lambda: signal.signal(
                    signal.SIGINT, signal.SIG_IGN),
            )
            futures = {}
            with pool:
                for k in tasks:
                    futures[pool.submit(_render_one_frame, k)] = k

                for i, fut in enumerate(as_completed(futures), 1):
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = f"FAIL worker: {e}"
                    print(f"  [{i}/{len(tasks)}] {result}")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if pool:
                pool.shutdown(wait=False, cancel_futures=True)
    else:
        try:
            for k in tasks:
                result = _render_one_frame(k)
                print(f"  {result}")
        except KeyboardInterrupt:
            print("\n  Interrupted.")

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

            out_video = os.path.join(
                data_dir, "vids", f"{series_base}{field}.mp4")
            os.makedirs(os.path.dirname(out_video), exist_ok=True)
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

    print(f"Done. Videos in {os.path.join(data_dir, 'vids')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D detonation VTKHDF render and video tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared parent for render and all commands
    _render_parent = argparse.ArgumentParser(add_help=False)
    _render_parent.add_argument("--series", required=True,
                                help="Path to .vtkhdf.series file or single .vtkhdf file")
    _render_parent.add_argument("--field", action="append", default=[],
                                help="Scalar field(s) to render (repeatable)")
    _render_parent.add_argument("-i", "--index", default="all",
                                help="Series indices: N, start:stop:step, or 'all'")
    _render_parent.add_argument("--step", type=str, default=None,
                                help="Match step number in filename")
    _render_parent.add_argument("--stride", type=int, default=None,
                                help="Render every Nth frame")
    _render_parent.add_argument("--render-width", type=int, default=5000,
                                help="Output image width in pixels (height auto from aspect)")
    _render_parent.add_argument("--render-dpi", type=int, default=72,
                                help="Screen-space DPI for font/tick sizing (higher = smaller)")
    _render_parent.add_argument("--hpad", type=float, default=1.04,
                                help="Horizontal padding factor")
    _render_parent.add_argument("--font-scale", type=float, default=2.0,
                                help="Base font scale multiplier")
    _render_parent.add_argument("--font-scale-time", type=float, default=1.0,
                                help="Time strip font scale (× base)")
    _render_parent.add_argument("--font-scale-cb", type=float, default=1.0,
                                help="Colorbar strip font scale (× base)")
    _render_parent.add_argument("--font-scale-axes", type=float, default=1.0,
                                help="Cube axes font + spacing scale (× base)")
    _render_parent.add_argument("--margin", type=float, default=0.03,
                                help="Canvas margin fraction")
    _render_parent.add_argument("--top-strip", type=float, default=0.12,
                                help="Top annotation strip height fraction")
    _render_parent.add_argument("--cb-strip", type=float, default=0.12,
                                help="Colorbar strip height fraction")
    _render_parent.add_argument("--render-x-lim", type=str, default=None,
                                help="Render region x-range: LOW:HIGH (default: full mesh)")
    _render_parent.add_argument("--render-y-lim", type=str, default=None,
                                help="Render region y-range: LOW:HIGH (default: full mesh)")
    _render_parent.add_argument("--clim-min", type=float, default=None,
                                help="Fixed colorbar minimum (applies to all fields)")
    _render_parent.add_argument("--clim-max", type=float, default=None,
                                help="Fixed colorbar maximum (applies to all fields)")
    _render_parent.add_argument("--clim", action="append", default=[],
                                help="Per-field range: FIELD=LOW:HIGH (e.g. T=300:4000 P=0:25)")
    _render_parent.add_argument("--clim-from", type=str, default=None,
                                help="Use this step number's data for colorbar range")
    _render_parent.add_argument("--cmap", type=str, default="plasma",
                                help="Colormap: plasma, inferno, viridis, turbo, hot, coolwarm, RdBu, magma, cividis, Blues, Reds, etc.")
    _render_parent.add_argument("--jobs", "-j", type=int, default=1,
                                help="Parallel workers (1 = serial, 0 = cpu_count)")
    _render_parent.add_argument("--global-clim", action="store_true",
                                help="Use global min/max across all selected frames")

    # Shared parent for combine and all commands
    _combine_parent = argparse.ArgumentParser(add_help=False)
    _combine_parent.add_argument("--fps", type=int, default=24,
                                 help="Frames per second")
    _combine_parent.add_argument("--crf", type=int, default=18,
                                 help="Video quality (lower = better, 18-28)")

    # ---- render ----
    p_render = sub.add_parser("render", help="Render VTKHDF frames to PNG",
                              parents=[_render_parent])

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
    p_all = sub.add_parser("all", help="Render frames then combine into video",
                           parents=[_render_parent, _combine_parent])

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
