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

### Creating white margins around the data

When the mesh extends beyond the region you want to show (e.g. a far-field
that reaches well past the body), you cannot get a clean white margin by
camera framing alone — zooming out just reveals more mesh. Options, in
order of preference:

1. **Give up the margin** (simplest, best for *mesh* plots). Frame the
   camera to the window (`parallel_scale = half window height`) and let the
   mesh fill the canvas; the frustum edges give a clean straight crop and
   real cell edges are preserved (no geometry modification).

2. **`clip_box` geometry clip** (good for *scalar field* plots). Clips the
   data to the window so white shows around it with `parallel_scale`
   padding:
   ```python
   box = sub.clip_box((x0, x1, y0, y1, z0, z1), invert=False)
   ```
   **Caveat:** it cuts cells, so quads/hexes become triangles. Harmless for
   filled `add_mesh(scalars=...)` colour, but it **ruins `show_edges` mesh
   plots** (spurious diagonal edges). Never use it for a mesh/grid figure.

3. **Layered multi-renderer** (faithful margin, preserves topology). Put a
   full-canvas white background renderer behind a data renderer placed in a
   sub-viewport. Tiled subplot renderers (see *Dedicated colorbar strip*)
   also avoid black bars because every renderer clears its area to white —
   unlike a single shrunk `SetViewport`, whose uncovered area renders black.
   Use this when you need both an unclipped mesh **and** a white margin.

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

### Dedicated colorbar strip (stable colorbar region)

Placing the scalar bar over the data with normalized `position_x/width/
height` fights VTK's internal layout: the title/labels spill outside the
bar box, proportions are hard to control, the bar or its background box
ends up too thick, and you need an opaque background panel
(`DrawBackgroundOn()` + frame) just to keep labels readable over the
field. **Don't fight it — give the colorbar its own renderer.**

Use a 2-renderer plotter (`shape=(2, 1)` for a bottom strip, or
`(1, 2)` for a side strip), override the viewports for an unequal split,
and put the bar in the thin strip on clean white. Because the two
renderers *tile the whole canvas*, every pixel is cleared to white — so
there are **no black bars** (unlike shrinking a single renderer's
viewport with `SetViewport`, where the uncovered area renders black).

```python
STRIP = 0.13   # bottom fraction reserved for the colorbar

pl = pv.Plotter(off_screen=True, window_size=[W, H],
                shape=(2, 1), border=False)   # border=False removes the
                                              # divider line between renderers

# --- data renderer (top) ---
pl.subplot(0, 0)
pl.background_color = "white"
actor = pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                    show_scalar_bar=False)     # no bar here
pl.camera.parallel_projection = True
pl.camera.focal_point = (cx, cy, 0.0)
pl.camera.position    = (cx, cy, 1.0)
pl.camera.view_up     = (0.0, 1.0, 0.0)
pl.camera.parallel_scale = (0.5 * domain_h) / 0.92   # ~4% margin

# --- colorbar renderer (thin bottom strip) ---
pl.subplot(1, 0)
pl.background_color = "white"
pl.add_scalar_bar(
    title=field, mapper=actor.mapper,   # reuse the data mesh's LUT
    vertical=False,
    position_x=0.20, position_y=0.55,   # coords are within THIS strip
    width=0.60, height=0.32,            # height = bar thickness; lower = thinner
    label_font_size=16, title_font_size=18, color="black", fmt="%.3g",
)

# unequal split: tall data renderer + thin colorbar strip
pl.renderers[0].SetViewport(0.0, STRIP, 1.0, 1.0)
pl.renderers[1].SetViewport(0.0, 0.0, 1.0, STRIP)
```

Notes:
- `mapper=actor.mapper` makes the bar in the second renderer use the data
  mesh's lookup table (color map + `clim`); without it the bar has no
  scalars to map.
- `border=False` on the `Plotter` removes the thin divider line VTK draws
  between subplot renderers.
- Expose `STRIP` and the bar `position/width/height` as constants — they
  are the proportions you will tune. The bar's *thickness* is its `height`
  (horizontal bar) and is the usual "too thick" culprit.
- This is the robust choice whenever you need a stable, non-overlapping
  colorbar region; the over-the-data overlay below is fine only for quick
  single-frame looks.

### Dedicated annotation strips (time text, titles)

Text overlays placed in the data renderer suffer the same problems as
overlaid colorbars: they can be invisible over the data colormap, position
drifts with camera framing, and there's no clean background.  **Give
annotations their own renderer strip.**  The pattern is identical to the
colorbar strip: a thin white renderer with only text actors.

Use a 4-renderer plotter with a full-canvas white background renderer
(see *Creating white margins*), then stack three strips on top:

```
[0] full-canvas white bg (fills gaps → no black bars)
[1] thin top strip — text-only, time/label
[2] data renderer — mesh + camera + axes
[3] thin bottom strip — colorbar
```

**Top strip — text-only with `add_text` at `lower_edge`:**

```python
# top strip: text is centred horizontally by VTK at lower_edge
pl.subplot(1, 0)
pl.background_color = "white"
pl.add_text(f"t = {time_val:.4e}", position="lower_edge",
            font_size=44, color="black")
pl.camera.parallel_projection = True   # suppress warning
```

Key points:
- `position="lower_edge"` — VTK centres the text horizontally and puts it
  at the bottom of this renderer's viewport (closer to the data strip
  below, not cut off at the very top).
- `position="upper_edge"` also works — text at the very top of the strip.
- Do NOT use `position=(x, y)` manually — the coordinate system with
  `SetViewport` overrides is confusing and unreliable for centering.
  Let VTK's built-in edge positions handle it.
- The top renderer needs `pl.camera.parallel_projection = True` only to
  suppress a VTK warning about an unset camera — no geometry is drawn.
- Strip height is typically 6% (`top_strip=0.06`) of canvas.

**Data renderer — mesh + axes:**

The data renderer goes in the middle. Axes are best rendered with
`show_bounds(location="all", ...)` which draws tick marks and labels on
all four edges of the renderer's viewport, adapted to the data bounds:

```python
pl.show_bounds(location="all", ticks="both", xtitle="x", ytitle="y",
               font_size=22, color="black", n_xlabels=5, n_ylabels=3)
```

If `show_bounds` labels are clipped or invisible (common with inset
viewports), try `location="all"` to show labels on all four sides, or
increase the camera's `h_pad_factor` (e.g. 1.08) to give the axis labels
room between the mesh edge and the viewport edge.

`show_bounds` returns a `CubeAxesActor`. Tweak its appearance after creation:

```python
ca = pl.show_bounds(...)
ca.SetScreenSize(10.0)       # overall size (higher = larger, default ~10)
ca.label_offset = 2.0        # text distance from tick (default 20 — huge)
ca.title_offset = (2.0, 5.0) # title distance from axis (default 20,20)
```

Tick length is not exposed through pyvista but `SetScreenSize` + small
`label_offset` produce compact, readable axes.

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
