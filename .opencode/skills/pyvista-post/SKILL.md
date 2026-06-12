---
name: pyvista-post
description: Use when rendering CFD VTKHDF output with pyvista. Covers camera control, layout design, colormaps, headless rendering, multiprocessing strategy, and video combination via ffmpeg. Trigger on pyvista, VTKHDF, render, screenshot, colormap, video, mp4, combine, or post-processing visualization.
license: MIT
compatibility: opencode
---

# PyVista Post-Processing Skill

Knowledge for rendering CFD VTKHDF output (2D/3D unstructured meshes)
with pyvista and combining frames into video. Covers camera setup,
layout, headless rendering, colormaps, and multiprocessing strategy.

---

## Headless Rendering

Use `EGL_PLATFORM=surfaceless` for off-screen rendering without X server:

```bash
EGL_PLATFORM=surfaceless python script.py
```

In Python:
```python
pv.OFF_SCREEN = True
pl = pv.Plotter(off_screen=True, window_size=[W, H])
```

Expect `bad X server connection. DISPLAY=` warnings — harmless on Linux
with EGL. To suppress, set `os.environ.setdefault("DISPLAY", ":0")`
before importing pyvista, but the warnings are benign.

## Camera Control

### Parallel (orthographic) projection

Required for 2D data (no perspective distortion):

```python
pl.camera.parallel_projection = True
pl.camera.position = (cx, cy, 1.0)       # camera location
pl.camera.focal_point = (cx, cy, 0.0)    # look-at point
pl.camera.view_up = (0.0, 1.0, 0.0)     # up direction
pl.camera.parallel_scale = ps            # half the visible height
```

### parallel_scale → visible region

The visible world-space region is computed from `parallel_scale` and
window aspect ratio:

```
visible_height = 2 * parallel_scale
visible_width  = visible_height * (window_width / window_height)
```

So the window aspect ratio directly controls the visible aspect ratio.
**For a domain of aspect ratio D:W to fill the canvas, the window must
also be D:W.**  A 10:1 domain needs a 10:1 window to fill it.

### Domain-as-strip layout (recommended for thin domains)

For domains much thinner than their length (e.g. 10:1 domain, 5:1
canvas), don't crop to the domain. Instead:

1. Use a canvas taller than the domain's natural aspect ratio
2. Set `parallel_scale` so the domain fills the full width with ~2-4%
   horizontal padding
3. The domain renders as a centered strip with white space above/below
4. Use the extra vertical space for colorbar, time text, annotations

```python
domain_w = mesh.bounds[1] - mesh.bounds[0]
h_pad_factor = 1.04  # 4% horizontal padding
ps = domain_w * h_pad_factor / (2 * (W / H))
```

**This is preferred over `SetViewport()`** because SetViewport creates
black bars outside the viewport.

### SetViewport — advanced sub-region rendering

```python
# Place renderer in a sub-region of the canvas (normalized coords)
pl.renderer.SetViewport(xmin, ymin, xmax, ymax)
```

Use only when you need multiple renderers. Avoid for single-domain
layouts — black bars outside the viewport are ugly.

### view_xy() helper

Sets camera looking down the +Z axis at the XY plane. Good for quick
visualization but doesn't fill the frame optimally:

```python
pl.view_xy()
pl.camera.zoom(1.2)  # adjust after
```

## Layout Control

### Window size

Always explicit — pyvista defaults are tiny:

```python
pl = pv.Plotter(off_screen=True, window_size=[5000, 1000])
```

### Background color

```python
pl.background_color = "white"
```

### Colorbar (scalar bar)

Positioned in viewport-normalized coordinates. Place in lower region
of a domain-as-strip layout:

```python
scalar_bar_args={
    "title": "T [K]",
    "vertical": False,           # horizontal bar
    "position_x": 0.2,           # center horizontally (0-1)
    "position_y": 0.08,          # near bottom
    "width": 0.6,                # 60% of viewport width
    "height": 0.06,              # 6% of viewport height
    "label_font_size": 48,       # large font for 5K-wide renders
    "title_font_size": 56,
    "fmt": "%.2g",               # use %.2g for mixed-scale fields
}
```

**Format string rules:**
- `"%.0f"` — integers only (bad for species fractions in [0,1])
- `"%.2g"` — 2 significant digits, auto-switches to scientific for large/small
- `"%.3e"` — always scientific, 3 decimal places

### Text overlays

```python
pl.add_text(f"t = {time_val:.4e}", position="upper_edge",
            font_size=44, color="black")
```

Positions: `"upper_edge"`, `"upper_right"`, `"lower_left"`, etc.
Non-dimensional time: use `:.4e` for scientific notation; remove `s`
unit since solver time is non-dimensional.

## Colormaps

pyvista uses matplotlib/VTK colormap names (string). Always specify
explicitly — never rely on defaults.

| Type | Good choices | Best for |
|------|-------------|----------|
| Perceptual sequential | `plasma`, `inferno`, `viridis`, `magma`, `cividis` | General scalar fields |
| High contrast / CFD | `turbo` | Fine gradients, shock structures |
| Temperature | `hot`, `afmhot` | Temperature fields (T) |
| Diverging | `coolwarm`, `RdBu`, `Spectral` | Signed quantities (e.g. velocity components) |
| Single hue | `Blues`, `Reds`, `Oranges` | Simple sequential |

```python
pl.add_mesh(mesh, scalars="T", cmap="inferno", clim=[300, 3900], ...)
```

**For species mass fractions** (`Y_*`): `viridis` or `inferno` with
linear scale. If the field spans many orders of magnitude (e.g. 1e-30
to 0.7), consider log scale (`log_scale=True`).

**For temperature**: `inferno` (dark→bright, intuitive for heat) or
`hot` (traditional).  Avoid `jet` (rainbow) — perceptually misleading.

## Color Range (clim)

By default, compute clim from the last frame in the batch (gives
consistent colors across all frames). Override with:

```python
# Fixed range
clim = (300.0, 4000.0)

# Global across all selected frames
clim = get_global_clim(data_dir, file_list, field)

# Per-frame (auto-scale — BAD for animations, causes pulsing)
clim = (data.min(), data.max())
```

**Always use consistent clim across all frames of a video.**
Per-frame auto-scaling makes the colorbar pulse and obscures the
evolution.

## Data Handling

### Reading VTKHDF

```python
mesh = pv.read("react_-T0__500.vtkhdf")
# mesh.bounds → (xmin, xmax, ymin, ymax, zmin, zmax)
# mesh.cell_data["T"] → scalar array
# mesh.n_cells, mesh.n_points
```

### 2D vs 3D data

- 2D mesh: `zmin == zmax == 0.0`, all points in XY plane
- Use `view_xy()` or manual parallel projection looking down +Z
- `view_up = (0, 1, 0)` puts Y up (standard Cartesian)
- 3D: same camera setup but `view_up` and `focal_point` need 3 coords

### Vector fields

If a field has shape `(N, 3)` it's a vector. For color-mapping,
compute magnitude:

```python
data = np.linalg.norm(mesh["Velo"], axis=1)
```

### Missing data / NaN handling

```python
vmin = float(np.nanmin(data))
vmax = float(np.nanmax(data))
```

## Multiprocessing

### Strategy: parallelize per mesh file, not per field

Each worker opens one VTKHDF file and renders all fields for that
file from a single `pv.read()`. This avoids redundant I/O and VTK
global state conflicts.

```python
from concurrent.futures import ProcessPoolExecutor

def render_one(args):
    fname, data_dir, fields, clims, ... = args
    mesh = pv.read(os.path.join(data_dir, fname))
    for field in fields:
        render_frame(mesh, field, ...)

with ProcessPoolExecutor(max_workers=N) as ex:
    futures = {ex.submit(render_one, t): t for t in tasks}
    for fut in as_completed(futures):
        result = fut.result()
```

**Important:**
- Use `ProcessPoolExecutor`, NOT `ThreadPoolExecutor` — VTK is not
  thread-safe and has global state
- Each worker MUST import pyvista and create its own Plotter
- Don't pass mesh objects between processes — pass filenames
- Compute color ranges (clim) in the main process before dispatching
- `-j 0` = use `os.cpu_count()` workers

### DO NOT parallelize across fields for the same file

Reading the same VTKHDF file multiple times from different processes
is wasteful. One read, all fields.

## Video Combination

### ffmpeg with symlinked sequence (recommended)

The most reliable approach. cv2's MP4 encoder is unreliable with
H.264 — use ffmpeg directly.

```python
import tempfile, subprocess

with tempfile.TemporaryDirectory() as tmpdir:
    for i, png_path in enumerate(ordered_pngs):
        dst = os.path.join(tmpdir, f"{i:06d}.png")
        os.symlink(os.path.abspath(png_path), dst)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",               # quality: 18 = visually lossless
        output_path,
    ]
    subprocess.run(cmd, check=True)
```

**Why symlinks instead of copying:** PNGs at 5000×1000 are ~100 KB
each. 251 frames × 4 fields = ~100 MB of copies avoided.

**Output naming:** Use the series prefix + field name with double
underscore, matching the VTKHDF naming convention:
- Series: `react_-T0_.vtkhdf.series`
- Output: `react_-T0_T.mp4`, `react_-T0_P.mp4`

### Ordering

Frames MUST be ordered by simulation time, not alphabetically by
filename. Read the `.vtkhdf.series` JSON and sort by `time`:

```python
files = sorted([(e["name"], e["time"]) for e in series["files"]],
               key=lambda x: x[1])
```

### Resolution preservation

ffmpeg with `-i %06d.png` preserves the exact PNG resolution. No
resize step needed. Verify with `ffprobe`:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames \
  -of csv=p=0 output.mp4
```

## Complete Render Function Template

```python
def render_frame(mesh, field, clim, time_val, out_path,
                 window_size=(5000, 1000), h_pad_factor=1.04,
                 font_scale=2.0, cmap="plasma"):
    W, H = window_size
    domain_w = mesh.bounds[1] - mesh.bounds[0]
    cx = (mesh.bounds[0] + mesh.bounds[1]) / 2
    cy = (mesh.bounds[2] + mesh.bounds[3]) / 2
    ps = domain_w * h_pad_factor / (2 * (W / H))

    pl = pv.Plotter(off_screen=True, window_size=[W, H])
    pl.background_color = "white"

    pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                show_edges=False,
                scalar_bar_args={
                    "title": field, "vertical": False,
                    "position_x": 0.2, "position_y": 0.08,
                    "width": 0.6, "height": 0.06,
                    "label_font_size": int(24 * font_scale),
                    "title_font_size": int(28 * font_scale),
                    "fmt": "%.2g",
                })

    pl.camera.parallel_projection = True
    pl.camera.position = (cx, cy, 1.0)
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.view_up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = ps

    pl.add_text(f"t = {time_val:.4e}", position="upper_edge",
                font_size=int(22 * font_scale), color="black")
    pl.screenshot(out_path)
    pl.close()
```

## Reference

Full working implementation (render + combine) bundled with this skill:
[render_detonation.py](render_detonation.py)
