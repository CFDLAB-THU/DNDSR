#!/usr/bin/env python3
"""Detonation shock speed analysis for DNDSR 1-D detonation output.

Usage:
    python workspace/detonation1d/analyze_detonation.py <output-dir> [<output-dir> ...]

Tracks the leading shock front (first cell where P > P_threshold * ambient)
through all VTU snapshots, fits the shock speed from a sliding 5-pt window,
and produces a position-vs-time plot.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import argparse
import importlib
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_HERE = _SCRIPT.parent
_IMPORT_ROOT = _HERE.parent.parent


matplotlib.use("Agg")

U0 = 379.0
P0 = U0 * U0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path,
                   help="DNDSR output directories")
    p.add_argument("--labels", nargs="*", help="Custom labels")
    p.add_argument("--out", type=Path, default=_HERE / "detonation_speed.png",
                   help="Output plot path")
    p.add_argument("--p-threshold", type=float, default=1.5,
                   help="Pressure ratio threshold for shock front (default 1.5 x ambient)")
    p.add_argument("--fit-pts", type=int, default=5,
                   help="Number of points for sliding-window fit (default 5)")
    p.add_argument("--t-max", type=float, default=1e100,
                   help="Max code time for fitting window")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _load_plot_digest():
    spec = importlib.util.spec_from_file_location(
        "plot_output_digest", _HERE.parent / "flame1d" / "plot_output_digest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shock_front(snap, p_thresh):
    """Return x-coordinate of the leading shock front (rightmost cell with P > threshold)."""
    x = np.asarray(snap["x"], float).reshape(-1)
    P = np.asarray(snap["cell_data"]["P"], float).reshape(-1)
    order = np.argsort(x)
    xs, Ps = x[order], P[order]
    P_bar = Ps * P0 / 101325.0
    shocked = P_bar > p_thresh
    if shocked.any():
        i = int(np.where(shocked)[0][-1])
        return float(xs[i])
    return None


def profile_at_snapshot(snap):
    """Return sorted (x, T(K), P(bar), u(m/s)) arrays for a VTU snapshot."""
    x = np.asarray(snap["x"], float).reshape(-1)
    T = np.asarray(snap["cell_data"]["T"], float).reshape(-1)
    P = np.asarray(snap["cell_data"]["P"], float).reshape(-1)
    v = np.asarray(snap["cell_data"]["Velo"], float).reshape((-1, 3))[:, 0]
    order = np.argsort(x)
    return x[order], T[order], P[order] * P0 / 1e5, v[order] * U0


def plot_profiles(pod, out_dir: Path, label: str, pod_label: str, out_path: Path,
                  n_snapshots: int = 6, dpi: int = 180):
    """Plot T, P, u profiles at evenly-spaced snapshots."""
    paths = sorted(
        [p for p in out_dir.glob("react___*.vtu")
         if not p.name.endswith("_C.vtu") and pod.vtu_step(p) is not None],
        key=lambda p: pod.vtu_step(p),
    )
    if len(paths) < 2:
        return

    indices = np.linspace(
        0, len(paths) - 1, min(n_snapshots, len(paths)), dtype=int)
    indices = np.unique(indices)
    selected = [paths[i] for i in indices]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(selected)))

    for i, p in enumerate(selected):
        s = pod.parse_vtu(p)
        xs, Ts, Ps, us = profile_at_snapshot(s)
        step = pod.vtu_step(p)
        lab = f"step={step} t={s['time']:.4f}"

        axes[0].plot(xs * 1e3, Ts, lw=1.2, color=colors[i], label=lab)
        axes[1].plot(xs * 1e3, Ps, lw=1.2, color=colors[i])
        axes[2].plot(xs * 1e3, us, lw=1.2, color=colors[i])

    axes[0].set_ylabel("T (K)")
    axes[1].set_ylabel("P (bar)")
    axes[2].set_ylabel("u (m/s)")
    axes[0].set_title(f"{pod_label} — Temperature")
    axes[1].set_title("Pressure")
    axes[2].set_title("Velocity")
    axes[0].legend(fontsize=7.5, ncols=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x (mm)")
    fig.suptitle(f"{label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Profile plot: {out_path}")


def sliding_fit(t_code, x_front, n_pts=5, t_max=1e100):
    """Slide an n_pts window through points with t_code < t_max,
    pick the best R² window from the later half."""
    eligible = t_code < t_max
    if eligible.sum() < n_pts:
        eligible = np.ones(len(t_code), dtype=bool)
    t_e = t_code[eligible]
    x_e = x_front[eligible]
    if len(t_e) < n_pts:
        return float("nan"), 0.0, t_e, x_e
    n_w = len(t_e) - n_pts + 1
    later = n_w // 2
    best_r2, best_s = -np.inf, later
    for s in range(later, n_w):
        tw, xw = t_e[s:s + n_pts], x_e[s:s + n_pts]
        c = np.polyfit(tw, xw, 1)
        xp = np.polyval(c, tw)
        ssr = np.sum((xw - xp) ** 2)
        sst = np.sum((xw - np.mean(xw)) ** 2)
        r2 = 1 - ssr / sst if sst > 1e-30 else 1.0
        if r2 > best_r2:
            best_r2, best_s = r2, s
    t_mid, x_mid = t_e[best_s:best_s + n_pts], x_e[best_s:best_s + n_pts]
    c = np.polyfit(t_mid, x_mid, 1)
    xp = np.polyval(c, t_mid)
    ssr = np.sum((x_mid - xp) ** 2)
    sst = np.sum((x_mid - np.mean(x_mid)) ** 2)
    r2 = 1 - ssr / sst if sst > 1e-30 else 1.0
    return c[0] * U0, r2, t_mid, x_mid


_COLORS = ["tab:blue", "tab:orange", "tab:cyan", "tab:red",
           "tab:green", "tab:purple"]


def main():
    args = parse_args()
    pod = _load_plot_digest()
    labels = args.labels or [d.name for d in args.dirs]
    if len(labels) != len(args.dirs):
        labels = [d.name[-30:] for d in args.dirs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for i, (d, label) in enumerate(zip(args.dirs, labels)):
        color = _COLORS[i % len(_COLORS)]
        paths = sorted(
            [p for p in d.glob("react___*.vtu")
             if not p.name.endswith("_C.vtu") and pod.vtu_step(p) is not None],
            key=lambda p: pod.vtu_step(p),
        )
        data = []
        for p in paths:
            s = pod.parse_vtu(p)
            fx = shock_front(s, args.p_threshold)
            if fx is not None:
                data.append((pod.vtu_step(p), float(s["time"]), fx))

        t_code = np.array([r[1] for r in data])
        x_front = np.array([r[2] for r in data])
        t_phys = t_code / U0 * 1e6  # μs

        speed, r2, t_mid, x_mid = sliding_fit(
            t_code, x_front, args.fit_pts, args.t_max)

        print(f"{label}: {len(data)} pts, speed={speed:.1f} m/s, R²={r2:.6f}")
        print(f"  start: x={x_front[0]*1e3:.3f}mm t={t_code[0]:.6f}")
        print(f"  end:   x={x_front[-1]*1e3:.3f}mm t={t_code[-1]:.6f}")

        ax1.plot(t_code, x_front * 1e3, "o-", ms=2,
                 lw=1, color=color, label=f"{label}")
        ax1.plot(t_mid, x_mid * 1e3, "s", ms=6, color=color, mec="k", mew=0.8)

        ax2.plot(t_phys, x_front * 1e3, "o-", ms=2, lw=1, color=color,
                 label=f"{label}: {speed:.0f} m/s" if np.isfinite(speed) else label)
        ax2.plot(t_mid / U0 * 1e6, x_mid * 1e3, "s",
                 ms=6, color=color, mec="k", mew=0.8)

        # Profile plot for this directory
        slug = label.replace(" ", "_").replace("/", "-")[:30]
        prof_path = _HERE / f"profiles_{slug}.png"
        plot_profiles(pod, d, f"{label} — profiles",
                      label, prof_path, dpi=args.dpi)

    ax1.set_xlabel("t (code units)")
    ax1.set_ylabel("shock x (mm)")
    ax1.set_title("Shock Front Position — fit points marked (□)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("t_phys (μs)")
    ax2.set_ylabel("shock x (mm)")
    ax2.set_title("Shock Front vs Physical Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
