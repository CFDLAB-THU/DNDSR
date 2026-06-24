"""
Orion capsule reentry: compare numerical wall heat flux (F4) with experiment.

Coordinate frames
-----------------
Experimental (American, left-hand): x = axial (toward back), y = lateral,
    z = upward.  Forebody data on symmetric plane (y_exp = 0).  Aftbody data
    at azimuthal angle phi, phi=0 on lateral side, phi=-90 on -z_exp side.
Numerical (right-hand, body-axial): x = axial, y = upward, z = lateral.
    Half model: only z_num >= 0 is simulated (symmetry plane z_num = 0).
    Geometry uses D = 10 (nondim); physical D = 0.254 m -> scale 0.0254.

exp -> num mapping (axes):  y_exp -> z_num (lateral),  z_exp -> y_num (upward).
Azimuthal angle about body axis x:  phi_num = atan2(y_num, z_num) == phi_exp.
    phi=0 -> +z_num (lateral), phi=+90 -> +y_num (up), phi=-90 -> -y_num (down).
Windward = -y_num (stagnation @ y~-3.84; freestream Velo = (cos28,sin28,0)).

Heat flux conversion
--------------------
F4 = energy-flux density through the wall, outward-from-fluid, nondim by
rho_ref*u_ref^3.  Wall heat INTO body is positive => q = -F4 * rho_ref*u_ref^3.
"""
import pyvista as pv
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Times-like math
})

# ----------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VTK = os.path.join(
    ROOT,
    "data/recv/Orion/redraw_run0/"
    "out-restart1-restart1-rot-rm8-cw-backin__1_20000_bnd.vtkhdf",
)
EXP_DIR = os.path.join(ROOT, "data/recv/Orion/Experimental")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# physical free-stream conditions (workspace/Orion/conditions.md)
GAMMA = 1.4
RGAS = 287.0  # J/(kg K), air
MA = 6.41
T_INF = 73.63  # K
P_INF = 4068.260149  # Pa

# geometry / normalization
NUM_D = 10.0          # nondim diameter used by the solver geometry
PHYS_D = 0.254        # m (10-in model)
SCALE = PHYS_D / NUM_D  # nondim length -> meters  (= 0.0254)

ZONE_FORE = 24
ZONE_AFT = 25

# convention toggles (tentative per problem statement; flip if debug plots disagree)
FORE_YEXP_SIGN = +1.0   # z_exp(up) -> FORE_YEXP_SIGN * y_num
AZI_SIGN = +1.0         # phi_num = AZI_SIGN * atan2(y_num, z_num)

FORE_Z_TOL = 0.07       # nondim band around symmetry plane for forebody line
AFT_PHI_TOL = 2.5       # deg band around target azimuth for aftbody line
# aftbody ends in a flat base cap (rounded corner -> cone). Exclude near-axial
# faces from the extracted line, and place markers on the side-facing radius.
# drop faces with |n_x| above this (perpendicular cap)
AFT_MAX_AXIAL_NORM = 0.97

# heat flux factor: q[W/m^2] = -F4 * Q_FACTOR
RHO_REF = P_INF / (RGAS * T_INF)
A_INF = np.sqrt(GAMMA * RGAS * T_INF)
U_REF = MA * A_INF
Q_FACTOR = RHO_REF * U_REF**3
print(
    f"rho_ref={RHO_REF:.6g} kg/m^3  a_inf={A_INF:.6g} m/s  u_ref={U_REF:.6g} m/s")
print(
    f"heat-flux factor rho_ref*u_ref^3 = {Q_FACTOR:.6g} W/m^2  (q = -F4*factor)")
# unit-Reynolds sanity check
mu = 1.716e-5 * (T_INF / 273.15) ** 1.5 * (273.15 + 110.4) / (T_INF + 110.4)
print(f"unit Re/m check = {RHO_REF*U_REF/mu:.4g}  (expect ~4.24e7)")


# ----------------------------------------------------------------------------- exp readers
def read_tecplot_point(path):
    """Return (ncols, data[N,ncols]) from a Tecplot POINT .dat file."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            toks = s.replace(",", " ").split()
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                continue
            if vals:
                rows.append(vals)
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


def load_forebody():
    d = read_tecplot_point(os.path.join(EXP_DIR, "Heat_Forebody.dat"))
    # cols: Y(lateral)=0, Z(normal/up), Q
    return {"z_up": d[:, 1], "q": d[:, 2]}


def load_aftbody():
    out = {}
    for fn in os.listdir(EXP_DIR):
        m = re.match(r"Heat_(-?\d+)_Aftbody\.dat", fn)
        if not m:
            continue
        phi = int(m.group(1))
        d = read_tecplot_point(os.path.join(EXP_DIR, fn))
        out[phi] = {"x": d[:, 0], "q": d[:, 1]}  # X(axial,m), Q
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------- numerical
def load_num():
    m = pv.read(VTK)
    cc = m.cell_centers().points
    fz = np.asarray(m.cell_data["FaceZone"])
    f4 = np.asarray(m.cell_data["F4"])
    nrm = np.asarray(m.cell_data["Norm"])
    nhat = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300)
    return m, cc, fz, f4, nhat


def zone_surface_tris(mesh, zone):
    """Triangulated surface (vertices in meters, triangle index array) for a FaceZone."""
    idx = np.where(np.asarray(mesh.cell_data["FaceZone"]) == zone)[0]
    surf = mesh.extract_cells(idx).extract_surface(
        nonlinear_subdivision=0).triangulate()
    pts = np.asarray(surf.points) * SCALE
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    return pts, faces


def num_forebody_line(cc, fz, f4):
    sel = (fz == ZONE_FORE) & (np.abs(cc[:, 2]) < FORE_Z_TOL)
    p = cc[sel]
    q = -f4[sel] * Q_FACTOR
    y_up = FORE_YEXP_SIGN * p[:, 1] * SCALE   # -> meters, exp-z(up) axis
    o = np.argsort(y_up)
    return y_up[o], q[o], p[o]


def num_aftbody_line(cc, fz, f4, nhat, phi_deg):
    sel = fz == ZONE_AFT
    p = cc[sel]
    n = nhat[sel]
    qy = AZI_SIGN * np.degrees(np.arctan2(p[:, 1], p[:, 2]))
    # drop perpendicular base-cap faces
    side = np.abs(n[:, 0]) < AFT_MAX_AXIAL_NORM
    band = (np.abs(qy - phi_deg) < AFT_PHI_TOL) & side
    pb = p[band]
    qb = -f4[sel][band] * Q_FACTOR
    x = pb[:, 0] * SCALE  # -> meters, axial
    o = np.argsort(x)
    return x[o], qb[o], pb[o]


# ----------------------------------------------------------------------------- sensor positions (debug)
def forebody_sensor_xyz(fore, cc, fz):
    """Place each forebody sensor on the body: y_num=z_up/scale, z=0, x from nearest cell."""
    p = cc[(fz == ZONE_FORE) & (np.abs(cc[:, 2]) < FORE_Z_TOL)]
    y_num = FORE_YEXP_SIGN * fore["z_up"] / SCALE
    xs, ys, zs = [], [], []
    for yv in y_num:
        j = np.argmin(np.abs(p[:, 1] - yv))
        xs.append(p[j, 0])
        ys.append(yv)
        zs.append(0.0)
    return np.column_stack([xs, ys, zs]) * SCALE


def aftbody_sensor_xyz(phi, xs_m, cc, fz, nhat):
    """Place aftbody sensors at the EXACT azimuth phi, using the side-facing
    surface radius r(x) (median of cap-excluded cells in a narrow x window).
    Exact phi keeps markers on clean radial meridians; excluding the flat base
    cap makes r(x) single-valued so the marker sits on the surface."""
    sel = fz == ZONE_AFT
    p = cc[sel]
    p = p[np.abs(nhat[sel][:, 0]) < AFT_MAX_AXIAL_NORM]   # drop flat base cap
    rcell = np.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
    x_num = xs_m / SCALE
    ph = np.radians(phi)
    out = []
    for xv in x_num:
        sel_x = np.zeros(len(p), dtype=bool)
        for tol in (0.06, 0.12, 0.25, 0.5):
            sel_x = np.abs(p[:, 0] - xv) < tol
            if sel_x.sum() >= 3:
                break
        if sel_x.sum() == 0:
            sel_x = np.argsort(np.abs(p[:, 0] - xv))[:20]
        r = np.median(rcell[sel_x])
        out.append([xv, r * np.sin(ph / AZI_SIGN), r * np.cos(ph / AZI_SIGN)])
    return np.array(out) * SCALE


# ----------------------------------------------------------------------------- plotting
def plot_forebody(fore, cc, fz, f4):
    yl, ql, _ = num_forebody_line(cc, fz, f4)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(yl, ql / 1e3, "-", color="C0", lw=1.5,
            label="Numerical (z$_{num}$=0)")
    ax.plot(fore["z_up"], fore["q"] / 1e3, "ks", ms=6, label="Experiment")
    ax.set_xlabel("upward coordinate  z$_{exp}$ = y$_{num}$ (m)")
    ax.set_ylabel("heat flux q (kW/m$^2$)")
    ax.set_title("Forebody symmetric line (windward = $-y$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forebody_line.png"), dpi=200)
    plt.close(fig)


def plot_aftbody(aft, cc, fz, f4, nhat):
    phis = list(aft.keys())
    ncol = 3
    nrow = int(np.ceil(len(phis) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(
        4.2 * ncol, 3.2 * nrow), squeeze=False)
    for k, phi in enumerate(phis):
        ax = axs[k // ncol][k % ncol]
        xl, ql, _ = num_aftbody_line(cc, fz, f4, nhat, phi)
        ax.plot(xl, ql / 1e3, "-", color="C1", lw=1.3, label="Numerical")
        ax.plot(aft[phi]["x"], aft[phi]["q"] / 1e3,
                "ks", ms=5, label="Experiment")
        ax.set_title(rf"Aftbody  $\varphi={phi}^\circ$")
        ax.set_xlabel("axial X (m)")
        ax.set_ylabel("q (kW/m$^2$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for k in range(len(phis), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aftbody_lines.png"), dpi=200)
    plt.close(fig)


def plot_sensor_projection(mesh, fore, aft, cc, fz, nhat):
    # body surfaces (pyvista triangulated) in meters
    fpts, ffac = zone_surface_tris(mesh, ZONE_FORE)
    apts, afac = zone_surface_tris(mesh, ZONE_AFT)
    fsens = forebody_sensor_xyz(fore, cc, fz)
    asens = {phi: aftbody_sensor_xyz(
        phi, aft[phi]["x"], cc, fz, nhat) for phi in aft}
    cmap = matplotlib.colormaps["turbo"]
    phis = list(aft.keys())
    colors = {phi: cmap((phi + 90) / 180.0) for phi in phis}

    def add_surface(ax, pts, fac, ia, ib, color):
        verts = pts[fac][:, :, [ia, ib]]
        ax.add_collection(
            PolyCollection(verts, facecolors=color, edgecolors="none", alpha=0.35,
                           zorder=0)
        )

    from matplotlib.patches import Patch
    surf_handles = [Patch(facecolor="#7fb3d5", alpha=0.35, label="forebody surf"),
                    Patch(facecolor="#f5b7a1", alpha=0.35, label="aftbody surf")]

    fig, (axyz, axxy) = plt.subplots(1, 2, figsize=(13, 6))
    # y-z projection (looking down the body axis): plot (z, y)
    add_surface(axyz, fpts, ffac, 2, 1, "#7fb3d5")
    add_surface(axyz, apts, afac, 2, 1, "#f5b7a1")
    h_fore = axyz.scatter(fsens[:, 2], fsens[:, 1], c="k", marker="*", s=110,
                          zorder=5, label="forebody sensors")
    aft_handles = []
    for phi in phis:
        s = asens[phi]
        h = axyz.scatter(s[:, 2], s[:, 1], color=colors[phi], s=50,
                         edgecolor="k", lw=0.4, zorder=4, label=f"aft {phi}")
        aft_handles.append(h)
    axyz.set_xlabel("z$_{num}$ lateral (m)")
    axyz.set_ylabel("y$_{num}$ up (m)")
    axyz.set_title(
        "y-z projection (down body axis)\nwindward=$-y$, sym plane z=0")
    axyz.set_aspect("equal")
    axyz.grid(alpha=0.3)
    axyz.autoscale_view()
    # x-y projection (side view): plot (x, y)
    add_surface(axxy, fpts, ffac, 0, 1, "#7fb3d5")
    add_surface(axxy, apts, afac, 0, 1, "#f5b7a1")
    axxy.scatter(fsens[:, 0], fsens[:, 1], c="k", marker="*", s=110, zorder=5)
    for phi in phis:
        s = asens[phi]
        axxy.scatter(s[:, 0], s[:, 1], color=colors[phi], s=50, edgecolor="k",
                     lw=0.4, zorder=4)
    axxy.set_xlabel("x$_{num}$ axial (m)")
    axxy.set_ylabel("y$_{num}$ up (m)")
    axxy.set_title("x-y projection (side view, flow $\\to$ +x, up = +y)")
    axxy.set_aspect("equal")
    axxy.grid(alpha=0.3)
    axxy.autoscale_view()
    # single unified legend, outside the axes (right margin)
    handles = surf_handles + [h_fore] + aft_handles
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.865, 0.5),
               fontsize=8, title="sensors / surfaces", frameon=True)
    fig.savefig(os.path.join(OUT, "sensor_projection.png"), dpi=200)
    plt.close(fig)


# order of segments along the aggregate axis: forebody, then aft +90 -> -90
AGG_AFT_ORDER = [90, 77, 64, 52, 26, 0, -45, -60, -90]


def plot_aggregate(fore, aft, cc, fz, f4, nhat):
    """One axis concatenating sensor lines: forebody -> aftbody 90 ... -90."""
    fig, ax = plt.subplots(figsize=(14, 5))
    offset = 0.0
    gap = 0.12  # fraction of mean segment width used as separator
    seg_widths = []
    # collect segments as (label, num_param, num_q, exp_param, exp_q)
    segs = []
    yl, ql, _ = num_forebody_line(cc, fz, f4)
    segs.append(("forebody", yl, ql, fore["z_up"], fore["q"]))
    for phi in AGG_AFT_ORDER:
        if phi not in aft:
            continue
        xl, qll, _ = num_aftbody_line(cc, fz, f4, nhat, phi)
        ex, eq = aft[phi]["x"], aft[phi]["q"]
        lo, hi = ex.min(), ex.max()
        pad = 0.05 * (hi - lo + 1e-9)
        # clip numerical to exp coverage
        msk = (xl >= lo - pad) & (xl <= hi + pad)
        segs.append((rf"$\varphi={phi}^\circ$", xl[msk], qll[msk], ex, eq))
    width = np.mean([max(s[1].max(), s[3].max()) - min(s[1].min(), s[3].min())
                     for s in segs])
    gap_abs = gap * width
    bounds = []
    centers = []
    first = True
    for lab, np_, nq, ep, eq in segs:
        pmin = min(np_.min(), ep.min())
        ax.plot(offset + (np_ - pmin), nq / 1e3, "-", color="C0", lw=1.3,
                label="Numerical" if first else None)
        ax.plot(offset + (ep - pmin), eq / 1e3, "ks", ms=5,
                label="Experiment" if first else None)
        first = False
        w = max(np_.max(), ep.max()) - pmin
        centers.append((offset + w / 2, lab))
        offset += w + gap_abs
        bounds.append(offset - gap_abs / 2)
    ymax = ax.get_ylim()[1]
    for cx, lab in centers:
        ax.text(cx, ymax * 0.98, lab, ha="center", va="top", fontsize=9)
    for b in bounds[:-1]:
        ax.axvline(b, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("concatenated sensor coordinate  "
                  "(forebody: y$_{up}$;  aftbody: axial X)  [segments left$\\to$right]")
    ax.set_ylabel("heat flux q (kW/m$^2$)")
    ax.set_title("Aggregate sensor heat flux:  forebody $\\to$ aftbody "
                 "$\\phi$=+90$\\to$-90")
    ax.set_xticks([])
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aggregate_sensors.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    fore = load_forebody()
    aft = load_aftbody()
    print("aftbody angles:", list(aft.keys()))
    mesh, cc, fz, f4, nhat = load_num()
    plot_forebody(fore, cc, fz, f4)
    plot_aftbody(aft, cc, fz, f4, nhat)
    plot_sensor_projection(mesh, fore, aft, cc, fz, nhat)
    plot_aggregate(fore, aft, cc, fz, f4, nhat)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
