"""
Orion symmetric-plane (z_num=0) temperature and pressure contours -- pyvista only.

FaceZone 22 of the boundary vtkhdf IS the z=0 symmetry plane, so it is rendered
directly as a coloured surface (real mesh cells -> the capsule stays a clean
void).  Numerical coords scaled to the physical model (D = 0.254 m), x-y window
~4D around the body.

Run headless:
    EGL_PLATFORM=surfaceless python workspace/Orion/orion_contour.py

T is physical Kelvin already.  P is nondim -> Pa via rho_ref*u_ref^2.
"""
import pyvista as pv
import numpy as np
import os

os.environ.setdefault("DISPLAY", ":0")

pv.OFF_SCREEN = True
# Times-like font for all text/scalar bars
pv.global_theme.font.family = "times"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VTK = os.path.join(
    ROOT,
    "data/recv/Orion/redraw_run0/"
    "out-restart1-restart1-rot-rm8-cw-backin__1_20000_bnd.vtkhdf",
)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# normalization (see orion_postproc.py / conditions.md)
GAMMA, RGAS, MA, T_INF, P_INF = 1.4, 287.0, 6.41, 73.63, 4068.260149
SCALE = 0.254 / 10.0
RHO_REF = P_INF / (RGAS * T_INF)
U_REF = MA * np.sqrt(GAMMA * RGAS * T_INF)
P_SCALE = RHO_REF * U_REF**2  # nondim P -> Pa

ZONE_SYM = 22
D = 0.254
XLIM = (-0.5 * D, 1.5 * D)   # 2D wide
YLIM = (-1.0 * D, 1.0 * D)   # 2D tall
# logical layout size (proportions/fonts tuned here)
W, H = 1000, 1100
IMG_SCALE = 3               # supersample factor at screenshot -> 3x pixel density
# data box = this fraction of visible height (~4% margin)
FILL_V = 0.92

# ---- colorbar layout (all easy to tune) -----------------------------------
STRIP = 0.13                 # bottom fraction of canvas reserved for the colorbar
CBAR_X = 0.20                # bar left   (fraction of strip width)
CBAR_Y = 0.55                # bar bottom (fraction of strip height)
CBAR_W = 0.60                # bar width  (fraction of strip width)
CBAR_H = 0.32                # bar thickness (fraction of strip height)
CBAR_LABEL_FS = 16
CBAR_TITLE_FS = 18


def render(sub, scalar, cmap, title, fname):
    pl = pv.Plotter(off_screen=True, window_size=[
                    W, H], shape=(2, 1), border=False)

    # --- data renderer (top) ---
    pl.subplot(0, 0)
    pl.background_color = "white"
    actor = pl.add_mesh(sub, scalars=scalar, cmap=cmap, show_edges=False,
                        show_scalar_bar=False)
    cx = 0.5 * (XLIM[0] + XLIM[1])
    cy = 0.5 * (YLIM[0] + YLIM[1])
    cam = pl.camera
    cam.parallel_projection = True
    cam.position = (cx, cy, 1.0)
    cam.focal_point = (cx, cy, 0.0)
    cam.view_up = (0.0, 1.0, 0.0)
    cam.parallel_scale = (0.5 * (YLIM[1] - YLIM[0])) / FILL_V

    # --- colorbar renderer (thin bottom strip, clean white background) ---
    pl.subplot(1, 0)
    pl.background_color = "white"
    pl.add_scalar_bar(
        title=title,
        mapper=actor.mapper,
        vertical=False,
        position_x=CBAR_X,
        position_y=CBAR_Y,
        width=CBAR_W,
        height=CBAR_H,
        label_font_size=CBAR_LABEL_FS,
        title_font_size=CBAR_TITLE_FS,
        color="black",
        fmt="%.3g",
    )

    # unequal split: tall data renderer + thin colorbar strip
    pl.renderers[0].SetViewport(0.0, STRIP, 1.0, 1.0)
    pl.renderers[1].SetViewport(0.0, 0.0, 1.0, STRIP)

    pl.screenshot(os.path.join(OUT, fname), scale=IMG_SCALE)
    pl.close()
    print("wrote", os.path.join(OUT, fname))


def render_mesh(sub, fname):
    """Mesh render: solid-colour surface with real cell edges (quad/tri), no
    clip_box -> actual mesh topology is visible.  Camera frames the window
    exactly (no white margin) because the mesh fills the canvas."""
    H_mesh = int(H * (1.0 - STRIP))
    pl = pv.Plotter(off_screen=True, window_size=[W, H_mesh])
    pl.background_color = "white"
    pl.add_mesh(sub, color="#cfe0f0", show_edges=True, edge_color="#1f3b57",
                line_width=0.5)
    cx = 0.5 * (XLIM[0] + XLIM[1])
    cy = 0.5 * (YLIM[0] + YLIM[1])
    pl.camera.parallel_projection = True
    pl.camera.position = (cx, cy, 1.0)
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.view_up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = 0.5 * (YLIM[1] - YLIM[0])  # FILL=1 -> no margin
    pl.screenshot(os.path.join(OUT, fname), scale=IMG_SCALE)
    pl.close()
    print("wrote", os.path.join(OUT, fname))


def main():
    m = pv.read(VTK)
    fz = np.asarray(m.cell_data["FaceZone"])
    sub = m.extract_cells(np.where(fz == ZONE_SYM)[0])
    sub.points = np.asarray(sub.points) * SCALE  # -> meters

    sub.cell_data["T [K]"] = np.asarray(sub.cell_data["T"])
    sub.cell_data["P [kPa]"] = np.asarray(sub.cell_data["P"]) * P_SCALE / 1e3

    # precise box clip -> clean straight edges (no ragged centroid extraction)
    box = sub.clip_box(
        (XLIM[0], XLIM[1], YLIM[0], YLIM[1], -1.0, 1.0), invert=False
    )
    print("clipped symmetry-plane cells:", box.n_cells)

    render(box, "T [K]", "inferno", "T [K]", "sym_temperature.png")
    render(box, "P [kPa]", "viridis", "P [kPa]", "sym_pressure.png")
    render_mesh(sub, "sym_mesh.png")   # full mesh, no clip -> real cell edges


if __name__ == "__main__":
    main()
