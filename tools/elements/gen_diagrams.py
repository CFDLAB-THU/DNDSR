#!/usr/bin/env python3
"""
Generate element topology diagrams.

For each element, produces multiple figures:
  1. *_nodes.png  — Node positions with indices (clean, no topology clutter)
  2. *_faces.png  — Subplot grid, one face per subplot highlighted on wireframe
  3. *_edges.png  — Subplot grid, edges grouped and highlighted on wireframe (3D only)

Usage:
    python3 -m tools.elements.gen_diagrams [--outdir DIR] [--format svg|png] [--element NAME]

Run from the project root directory.
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tools.elements.element_data import ALL_ELEMENTS, ElementData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGHLIGHT_FACE_FILL = (0.2, 0.55, 1.0, 0.30)
HIGHLIGHT_EDGE_COLOR = "#c0392b"
NORMAL_ARROW_COLOR = "#8B0000"
WIREFRAME_LINE = "silver"
GHOST_NODE_COLOR = "#aaaaaa"
DPI = 200

# Per-element-class 3D view angles for clarity
VIEW_ANGLES = {
    "TetSpace":     (30, -50),
    "HexSpace":     (22, -48),
    "PrismSpace":   (18, -70),
    "PyramidSpace": (12, -35),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coords(e: ElementData):
    return np.array(e.node_coords_0based(), dtype=np.float64)


def _face_nv(ft: str):
    if "Tri" in ft:
        return 3
    if "Quad" in ft:
        return 4
    return 2


def _centroid(coords, nv):
    return coords[:nv].mean(axis=0)


def _fnormal_3d(coords, fn):
    v0, v1, v2 = coords[fn[0]], coords[fn[1]], coords[fn[2]]
    n = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n)
    return n / mag if mag > 1e-15 else n


def _fnormal_2d(coords, fn):
    v0, v1 = coords[fn[0]], coords[fn[1]]
    t = v1 - v0
    n = np.array([t[1], -t[0], 0.0])
    mag = np.linalg.norm(n)
    return n / mag if mag > 1e-15 else n


def _set_3d(ax, coords_v, e):
    lo = coords_v.min(0)
    hi = coords_v.max(0)
    ranges = hi - lo
    mid = (lo + hi) * 0.5

    # Equal range on all axes (no geometric distortion)
    r = max(ranges) * 0.65
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x", fontsize=7, labelpad=1)
    ax.set_ylabel("y", fontsize=7, labelpad=1)
    ax.set_zlabel("z", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5, pad=0)
    elev, azim = VIEW_ANGLES.get(e.param_space, (25, -55))
    ax.view_init(elev=elev, azim=azim)


def _set_2d(ax, coords, pad=0.2):
    xmin, xmax = coords[:, 0].min() - pad, coords[:, 0].max() + pad
    ymin, ymax = coords[:, 1].min() - pad, coords[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=6)


def _grid(n, max_cols=3):
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    return rows, cols


# ---------------------------------------------------------------------------
# Wireframe
# ---------------------------------------------------------------------------

def _wire_2d(ax, e, c):
    for fn in e.face_nodes_0based():
        v0, v1 = fn[0], fn[1]
        ax.plot([c[v0, 0], c[v1, 0]], [c[v0, 1], c[v1, 1]],
                color=WIREFRAME_LINE, lw=1, zorder=1)


def _wire_3d(ax, e, c):
    for en in e.edge_nodes_0based():
        v0, v1 = en[0], en[1]
        ax.plot([c[v0, 0], c[v1, 0]], [c[v0, 1], c[v1, 1]], [c[v0, 2], c[v1, 2]],
                color=WIREFRAME_LINE, lw=0.8, zorder=1)


# ---------------------------------------------------------------------------
# Node drawing
# ---------------------------------------------------------------------------

def _label_offset_2d(coords, i, centroid):
    """Push label away from centroid."""
    d = coords[i, :2] - centroid[:2]
    mag = np.linalg.norm(d)
    if mag < 1e-12:
        return np.array([0.06, 0.06])
    return d / mag * 0.10


def _nodes_2d(ax, e, c, fs=9, centroid=None):
    if centroid is None:
        centroid = _centroid(c, e.num_vertices)
    for i in range(e.num_nodes):
        m = "o" if i < e.num_vertices else "s"
        s = 50 if i < e.num_vertices else 28
        clr = "black" if i < e.num_vertices else "dimgray"
        ax.scatter(c[i, 0], c[i, 1], s=s, c=clr, zorder=5, marker=m)
        off = _label_offset_2d(c, i, centroid)
        ax.text(c[i, 0] + off[0], c[i, 1] + off[1], str(i),
                fontsize=fs, ha="center", va="center", zorder=6,
                fontweight="bold" if i < e.num_vertices else "normal")


def _nodes_3d(ax, e, c, fs=8):
    for i in range(e.num_nodes):
        m = "o" if i < e.num_vertices else "s"
        s = 40 if i < e.num_vertices else 18
        clr = "black" if i < e.num_vertices else "dimgray"
        ax.scatter(c[i, 0], c[i, 1], c[i, 2], s=s, c=clr, zorder=5,
                   marker=m, depthshade=False)
        ax.text(c[i, 0], c[i, 1], c[i, 2], f"  {i}", fontsize=fs, zorder=6,
                fontweight="bold" if i < e.num_vertices else "normal")


def _ghost_nodes_2d(ax, e, c, exclude=set()):
    for i in range(e.num_nodes):
        if i in exclude:
            continue
        ax.scatter(c[i, 0], c[i, 1], s=18, c=GHOST_NODE_COLOR, zorder=3)


def _ghost_nodes_3d(ax, e, c, exclude=set()):
    for i in range(e.num_vertices):
        if i in exclude:
            continue
        ax.scatter(c[i, 0], c[i, 1], c[i, 2], s=15, c=GHOST_NODE_COLOR,
                   zorder=3, depthshade=False)
        ax.text(c[i, 0], c[i, 1], c[i, 2], f"  {i}",
                fontsize=6, color="#999999", zorder=3)


# ---------------------------------------------------------------------------
# 1. Node diagram
# ---------------------------------------------------------------------------

def gen_nodes(e, c, outdir, fmt):
    if e.dim == 1:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.plot([c[0, 0], c[1, 0]], [0, 0], "k-", lw=2)
        for i in range(e.num_nodes):
            m = "o" if i < e.num_vertices else "s"
            s = 60 if i < e.num_vertices else 35
            clr = "black" if i < e.num_vertices else "gray"
            ax.scatter(c[i, 0], 0, s=s, c=clr, zorder=5, marker=m)
            ax.text(c[i, 0], 0.07, str(i), fontsize=10, ha="center", va="bottom",
                    fontweight="bold" if i < e.num_vertices else "normal")
        ax.set_ylim(-0.2, 0.3)
        ax.set_aspect("equal")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    elif e.dim == 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        _wire_2d(ax, e, c)
        _nodes_2d(ax, e, c, fs=10)
        _set_2d(ax, c, pad=0.25)
    else:
        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        _wire_3d(ax, e, c)
        _nodes_3d(ax, e, c, fs=9)
        _set_3d(ax, c[:e.num_vertices], e)

    fig.suptitle(f"{e.dndsr_name} ({e.cgns_name}) — Nodes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, f"{e.dndsr_name}_nodes.{fmt}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Face diagrams
# ---------------------------------------------------------------------------

def _face_2d(ax, e, c, fi, fn):
    _wire_2d(ax, e, c)
    cent = _centroid(c, e.num_vertices)

    # Highlight edge
    v0, v1 = fn[0], fn[1]
    ax.plot([c[v0, 0], c[v1, 0]], [c[v0, 1], c[v1, 1]],
            color=HIGHLIGHT_EDGE_COLOR, lw=3.5, zorder=3)

    # Normal arrow (quiver for reliability)
    mid = (c[v0] + c[v1]) * 0.5
    n = _fnormal_2d(c, fn)
    if np.dot(n, mid - cent) < 0:
        n = -n
    scale = 0.30
    ax.quiver(mid[0], mid[1], n[0] * scale, n[1] * scale,
              angles="xy", scale_units="xy", scale=1,
              color=NORMAL_ARROW_COLOR, width=0.012, zorder=4)

    # Label face nodes
    for ni in fn:
        ax.scatter(c[ni, 0], c[ni, 1], s=45, c=HIGHLIGHT_EDGE_COLOR,
                   zorder=6, marker="o" if ni < e.num_vertices else "s")
        off = _label_offset_2d(c, ni, cent)
        ax.text(c[ni, 0] + off[0], c[ni, 1] + off[1], str(ni),
                fontsize=8, color=HIGHLIGHT_EDGE_COLOR, fontweight="bold", zorder=7)

    _ghost_nodes_2d(ax, e, c, exclude=set(fn))
    _set_2d(ax, c, pad=0.35)
    ax.set_title(f"F{fi}: {e.face_types[fi]} [{', '.join(str(n) for n in fn)}]",
                 fontsize=9, pad=4)


def _face_3d(ax, e, c, fi, fn):
    _wire_3d(ax, e, c)
    cent = _centroid(c, e.num_vertices)

    nv = _face_nv(e.face_types[fi])
    verts = [c[fn[j]] for j in range(nv)]

    poly = Poly3DCollection([verts], alpha=HIGHLIGHT_FACE_FILL[3],
                            facecolors=[HIGHLIGHT_FACE_FILL[:3]],
                            edgecolors=[HIGHLIGHT_EDGE_COLOR], linewidths=1.8)
    ax.add_collection3d(poly)

    # Normal arrow
    fc = np.mean(verts, axis=0)
    n = _fnormal_3d(c, fn)
    if np.dot(n, fc - cent) < 0:
        n = -n
    ax.quiver(fc[0], fc[1], fc[2], n[0] * 0.7, n[1] * 0.7, n[2] * 0.7,
              color=NORMAL_ARROW_COLOR, arrow_length_ratio=0.18, linewidth=2.5)

    # Highlighted nodes
    for ni in fn[:nv]:
        ax.scatter(c[ni, 0], c[ni, 1], c[ni, 2], s=35, c=HIGHLIGHT_EDGE_COLOR,
                   zorder=6, depthshade=False)
        ax.text(c[ni, 0], c[ni, 1], c[ni, 2], f"  {ni}",
                fontsize=8, color=HIGHLIGHT_EDGE_COLOR, fontweight="bold", zorder=7)

    _ghost_nodes_3d(ax, e, c, exclude=set(fn[:nv]))
    _set_3d(ax, c[:e.num_vertices], e)

    node_str = ', '.join(str(n) for n in fn[:nv])
    if len(fn) > nv:
        node_str += f" +{len(fn) - nv}mid"
    ax.set_title(f"F{fi}: {e.face_types[fi]} [{node_str}]", fontsize=8, pad=2)


def gen_faces(e, c, outdir, fmt):
    if e.num_faces == 0:
        return None

    fn0 = e.face_nodes_0based()
    nf = e.num_faces
    rows, cols = _grid(nf, max_cols=3)

    if e.dim <= 2:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_flat = np.atleast_1d(axes).flatten().tolist()
    else:
        fig = plt.figure(figsize=(4.5 * cols, 4.2 * rows))
        axes_flat = [fig.add_subplot(rows, cols, i + 1, projection="3d")
                     for i in range(rows * cols)]

    for fi in range(nf):
        ax = axes_flat[fi]
        if e.dim <= 2:
            _face_2d(ax, e, c, fi, fn0[fi])
        else:
            _face_3d(ax, e, c, fi, fn0[fi])

    for i in range(nf, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"{e.dndsr_name} — Faces ({nf})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2, w_pad=1.5)
    path = os.path.join(outdir, f"{e.dndsr_name}_faces.{fmt}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Edge diagrams (3D only, using edge_groups from element data)
# ---------------------------------------------------------------------------

EDGE_GROUP_COLORS = [
    "#c0392b", "#27ae60", "#2980b9", "#e67e22",
]


def gen_edges(e, c, outdir, fmt):
    if e.num_edges == 0:
        return None

    en0 = e.edge_nodes_0based()

    # Use edge_groups if defined, otherwise fall back to chunks of 3
    if e.edge_groups:
        groups = [(label, indices) for label, indices in e.edge_groups]
    else:
        ne = e.num_edges
        per_sub = 3
        groups = [(f"E{s}-E{min(s + per_sub, ne) - 1}",
                   list(range(s, min(s + per_sub, ne))))
                  for s in range(0, ne, per_sub)]

    n_sub = len(groups)
    rows, cols = _grid(n_sub, max_cols=3)

    fig = plt.figure(figsize=(4.5 * cols, 4.2 * rows))
    axes_flat = [fig.add_subplot(rows, cols, i + 1, projection="3d")
                 for i in range(rows * cols)]

    for si, (group_label, edge_indices) in enumerate(groups):
        ax = axes_flat[si]
        _wire_3d(ax, e, c)

        for gi, ei in enumerate(edge_indices):
            en = en0[ei]
            v0, v1 = en[0], en[1]
            clr = EDGE_GROUP_COLORS[gi % len(EDGE_GROUP_COLORS)]

            # Draw thick edge
            ax.plot([c[v0, 0], c[v1, 0]], [c[v0, 1], c[v1, 1]], [c[v0, 2], c[v1, 2]],
                    color=clr, lw=3, zorder=3)

            # Endpoint markers + labels
            for ni in [v0, v1]:
                ax.scatter(c[ni, 0], c[ni, 1], c[ni, 2], s=40, c=clr,
                           zorder=6, depthshade=False)
                ax.text(c[ni, 0], c[ni, 1], c[ni, 2], f"  {ni}",
                        fontsize=7, color=clr, fontweight="bold", zorder=7)

            # Mid-edge node for O2
            if len(en) == 3:
                nm = en[2]
                ax.scatter(c[nm, 0], c[nm, 1], c[nm, 2], s=25, c=clr,
                           zorder=6, marker="s", depthshade=False)
                ax.text(c[nm, 0], c[nm, 1], c[nm, 2], f"  {nm}",
                        fontsize=6, color=clr, zorder=7)

            # E-label at edge midpoint with background box
            mid = (c[v0] + c[v1]) * 0.5
            ax.text(mid[0], mid[1], mid[2], f" E{ei}",
                    fontsize=9, color=clr, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=clr, alpha=0.85, linewidth=0.8))

        _ghost_nodes_3d(ax, e, c, exclude=set())
        _set_3d(ax, c[:e.num_vertices], e)

        # Subplot title: group label + edge list
        parts = [f"E{ei}" for ei in edge_indices]
        ax.set_title(f"{group_label}: {', '.join(parts)}", fontsize=8, pad=2)

    for i in range(n_sub, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"{e.dndsr_name} — Edges ({e.num_edges})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2, w_pad=1.5)
    path = os.path.join(outdir, f"{e.dndsr_name}_edges.{fmt}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(e, outdir, fmt):
    c = _coords(e)
    paths = [gen_nodes(e, c, outdir, fmt)]
    p = gen_faces(e, c, outdir, fmt)
    if p:
        paths.append(p)
    p = gen_edges(e, c, outdir, fmt)
    if p:
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate element topology diagrams")
    parser.add_argument("--outdir", "-o", type=str,
                        default=os.path.join(PROJECT_ROOT, "docs", "elements"),
                        help="Output directory")
    parser.add_argument("--format", "-f", type=str, default="png",
                        choices=["svg", "png"], help="Output format")
    parser.add_argument("--element", "-e", type=str, default=None,
                        help="Generate only the named element")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    elements = ALL_ELEMENTS
    if args.element:
        elements = [el for el in ALL_ELEMENTS if el.dndsr_name == args.element]
        if not elements:
            names = [el.dndsr_name for el in ALL_ELEMENTS]
            print(f"Error: unknown '{args.element}'. Available: {names}", file=sys.stderr)
            sys.exit(1)

    total = 0
    for el in elements:
        paths = generate_all(el, args.outdir, args.format)
        for p in paths:
            print(f"  {os.path.relpath(p, PROJECT_ROOT)}")
        total += len(paths)

    print(f"\nDone: {total} diagrams for {len(elements)} elements "
          f"in {os.path.relpath(args.outdir, PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
