#!/usr/bin/env python3
"""
Generate element topology diagrams.

For each element, produces an SVG/PNG showing:
  - Node positions with indices
  - Edges highlighted and labeled (3D elements)
  - Faces highlighted with transparent fill and labeled
  - Outward face normal arrows at face centroids

Usage:
    python3 -m tools.elements.gen_diagrams [--outdir DIR] [--format svg|png]

Run from the project root directory.
"""

import argparse
import os
import sys
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tools.elements.element_data import ALL_ELEMENTS, ElementData

# Color palettes
FACE_COLORS = [
    (0.2, 0.6, 1.0, 0.15),   # blue
    (1.0, 0.4, 0.4, 0.15),   # red
    (0.3, 0.8, 0.3, 0.15),   # green
    (1.0, 0.8, 0.2, 0.15),   # yellow
    (0.7, 0.3, 0.9, 0.15),   # purple
    (1.0, 0.6, 0.2, 0.15),   # orange
]

EDGE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def _coords_0based(e: ElementData):
    """Return node coordinates as (N, 3) numpy float64 array, 0-based."""
    coords = e.node_coords_0based()
    return np.array(coords, dtype=np.float64)


def _face_centroid(coords, face_nodes_0):
    """Centroid of face vertex positions."""
    # Use only the first min(len, numVertices-of-face) nodes for centroid
    # For simplicity, average all listed nodes
    pts = coords[face_nodes_0]
    return pts.mean(axis=0)


def _face_normal_3d(coords, face_nodes_0):
    """Outward normal of a 3D face from first 3 vertices."""
    v0 = coords[face_nodes_0[0]]
    v1 = coords[face_nodes_0[1]]
    v2 = coords[face_nodes_0[2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm > 1e-15:
        n /= norm
    return n


def _face_normal_2d(coords, face_nodes_0):
    """Outward normal of a 2D face (edge) in the xy-plane."""
    v0 = coords[face_nodes_0[0]]
    v1 = coords[face_nodes_0[1]]
    tangent = v1 - v0
    n = np.array([tangent[1], -tangent[0], 0.0])
    norm = np.linalg.norm(n)
    if norm > 1e-15:
        n /= norm
    return n


def draw_element_2d(e: ElementData, ax):
    """Draw a 2D element with faces (edges) and nodes."""
    coords = _coords_0based(e)
    centroid = coords[:e.num_vertices].mean(axis=0)

    face_nodes_0 = e.face_nodes_0based()

    # Draw faces (edges) as colored lines
    for fi, fn in enumerate(face_nodes_0):
        # Only use the first 2 nodes (vertices) for the line
        verts = fn[:2]
        color = EDGE_COLORS[fi % len(EDGE_COLORS)]
        xs = [coords[verts[0], 0], coords[verts[1], 0]]
        ys = [coords[verts[0], 1], coords[verts[1], 1]]
        ax.plot(xs, ys, color=color, linewidth=2.5, zorder=2)

        # Face label at midpoint
        mid = (coords[verts[0]] + coords[verts[1]]) * 0.5
        n = _face_normal_2d(coords, fn)
        # Ensure outward
        if np.dot(n, mid - centroid) < 0:
            n = -n
        label_pos = mid[:2] + n[:2] * 0.12
        ax.text(label_pos[0], label_pos[1], f"F{fi}",
                fontsize=7, ha="center", va="center", color=color, fontweight="bold")

        # Normal arrow
        ax.annotate("", xy=(mid[0] + n[0] * 0.15, mid[1] + n[1] * 0.15),
                     xytext=(mid[0], mid[1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # Draw nodes
    for i in range(e.num_nodes):
        marker = "o" if i < e.num_vertices else "s"
        size = 50 if i < e.num_vertices else 30
        color = "black" if i < e.num_vertices else "gray"
        ax.scatter(coords[i, 0], coords[i, 1], s=size, c=color, zorder=5, marker=marker)
        offset = np.array([0.06, 0.06])
        ax.text(coords[i, 0] + offset[0], coords[i, 1] + offset[1],
                str(i), fontsize=8, ha="left", va="bottom", zorder=6)

    ax.set_aspect("equal")
    ax.set_title(f"{e.dndsr_name} ({e.cgns_name})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def draw_element_3d(e: ElementData, ax):
    """Draw a 3D element with faces, edges, and nodes."""
    coords = _coords_0based(e)
    centroid = coords[:e.num_vertices].mean(axis=0)

    face_nodes_0 = e.face_nodes_0based()
    edge_nodes_0 = e.edge_nodes_0based()

    # Draw faces as transparent polygons
    for fi, fn in enumerate(face_nodes_0):
        # Use only vertex nodes for the polygon
        # Determine how many are vertices: for the face element type
        ft = e.face_types[fi]
        if "Tri" in ft:
            nv = 3
        elif "Quad" in ft:
            nv = 4
        else:
            nv = len(fn)
        verts = [coords[fn[j]] for j in range(nv)]
        color = FACE_COLORS[fi % len(FACE_COLORS)]
        edge_color = list(color[:3]) + [0.6]

        poly = Poly3DCollection([verts], alpha=color[3],
                                facecolors=[color[:3]],
                                edgecolors=[edge_color], linewidths=0.8)
        ax.add_collection3d(poly)

        # Face label at centroid
        fc = _face_centroid(coords, fn[:nv])
        n = _face_normal_3d(coords, fn)
        if np.dot(n, fc - centroid) < 0:
            n = -n
        label_pos = fc + n * 0.15
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f"F{fi}", fontsize=7, ha="center", va="center",
                color=list(color[:3]), fontweight="bold")

    # Draw edges as colored lines
    for ei, en in enumerate(edge_nodes_0):
        v0, v1 = en[0], en[1]
        color = EDGE_COLORS[ei % len(EDGE_COLORS)]
        xs = [coords[v0, 0], coords[v1, 0]]
        ys = [coords[v0, 1], coords[v1, 1]]
        zs = [coords[v0, 2], coords[v1, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, zorder=2)

        # Edge label at midpoint
        mid = (coords[v0] + coords[v1]) * 0.5
        ax.text(mid[0], mid[1], mid[2], f"E{ei}",
                fontsize=5, ha="center", va="center", color=color, alpha=0.8)

    # Draw nodes
    for i in range(e.num_nodes):
        marker = "o" if i < e.num_vertices else "s"
        size = 40 if i < e.num_vertices else 20
        color = "black" if i < e.num_vertices else "dimgray"
        ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                   s=size, c=color, zorder=5, marker=marker, depthshade=False)
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                f" {i}", fontsize=7, zorder=6)

    ax.set_title(f"{e.dndsr_name} ({e.cgns_name})", fontsize=11, fontweight="bold")

    # Equal aspect for 3D
    all_pts = coords[:e.num_vertices]
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = max(ranges) * 0.6
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def draw_element_1d(e: ElementData, ax):
    """Draw a 1D element with nodes."""
    coords = _coords_0based(e)

    # Draw the line
    ax.plot([coords[0, 0], coords[1, 0]], [0, 0], "k-", linewidth=2.5, zorder=2)

    # Draw nodes
    for i in range(e.num_nodes):
        marker = "o" if i < e.num_vertices else "s"
        size = 60 if i < e.num_vertices else 35
        color = "black" if i < e.num_vertices else "gray"
        ax.scatter(coords[i, 0], 0, s=size, c=color, zorder=5, marker=marker)
        ax.text(coords[i, 0], 0.05, str(i), fontsize=9, ha="center", va="bottom", zorder=6)

    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"{e.dndsr_name} ({e.cgns_name})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])


def generate_diagram(e: ElementData, outdir: str, fmt: str = "svg"):
    """Generate a single element diagram."""
    if e.dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))
        draw_element_1d(e, ax)
    elif e.dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        draw_element_2d(e, ax)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_element_3d(e, ax)

    fig.tight_layout()
    path = os.path.join(outdir, f"{e.dndsr_name}.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate element topology diagrams")
    parser.add_argument("--outdir", "-o", type=str,
                        default=os.path.join(PROJECT_ROOT, "docs", "elements"),
                        help="Output directory for diagrams")
    parser.add_argument("--format", "-f", type=str, default="svg",
                        choices=["svg", "png"], help="Output format")
    parser.add_argument("--element", "-e", type=str, default=None,
                        help="Generate only the named element")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    elements = ALL_ELEMENTS
    if args.element:
        elements = [e for e in ALL_ELEMENTS if e.dndsr_name == args.element]
        if not elements:
            names = [e.dndsr_name for e in ALL_ELEMENTS]
            print(f"Error: unknown element '{args.element}'. Available: {names}",
                  file=sys.stderr)
            sys.exit(1)

    for e in elements:
        path = generate_diagram(e, args.outdir, args.format)
        print(f"  {e.dndsr_name} -> {os.path.relpath(path, PROJECT_ROOT)}")

    print(f"\nDone: {len(elements)} diagrams generated in {os.path.relpath(args.outdir, PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
