#!/usr/bin/env python3
"""Flame speed analysis and marker-position plotting for DNDSR output.

Usage:
    python workspace/flame1d/analyze_flame_speed.py <output-dir> [<output-dir> ...]

Each <output-dir> is a DNDSR solver output directory containing react___*.vtu
and react__*.log files.  The script:

1. Tracks the midpoint-temperature front through all VTU snapshots.
2. Fits the marker speed from a 5-point window in the middle of the
   free-propagation phase (R² > 0.999).
3. Measures the unburned gas velocity ahead of the flame at the
   centre of the fitting window.
4. Computes Su = marker_speed − u_unburned.
5. Writes a combined front-position plot and prints a summary table.

Cantera reference is taken from project script
scripts/reaction/h2o2_free_flame_mixture_averaged.py
(Su = 2.2540 m/s for stoichiometric H2/air).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import argparse
import importlib
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_HERE = _SCRIPT.parent
_IMPORT_ROOT = _HERE.parent.parent  # project root


matplotlib.use("Agg")


CANTERA_REF_SU = 2.25396598   # from project cantera script
CANTERA_T_BURNED = 2360.016463669699
CANTERA_T_FRESH = 300.0
U0 = 379.0                   # code-unit velocity scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path,
                   help="DNDSR output directories")
    p.add_argument("--labels", nargs="*",
                   help="Custom labels (same order as dirs)")
    p.add_argument("--out", type=Path, default=_HERE / "flame_speed_analysis.png",
                   help="Output plot path")
    p.add_argument("--t-mid", type=float, default=None,
                   help="Midpoint temperature threshold (default: 0.5*(T_fresh+T_burned))")
    p.add_argument("--fit-pts", type=int, default=5,
                   help="Number of mid-range points for linear fit (default 5)")
    p.add_argument("--u0", type=float, default=U0,
                   help="Code-unit velocity scale")
    p.add_argument("--su-ref", type=float, default=CANTERA_REF_SU,
                   help="Cantera reference Su for ratio column")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


# ---- VTU helpers (lightweight copies from plot_output_digest) ----

def _load_plot_digest():
    spec = importlib.util.spec_from_file_location(
        "plot_output_digest", _HERE / "plot_output_digest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Data extraction ----

def extract_front_data(pod, out_dir: Path, t_mid: float):
    """Return list of (step, t_code, front_x) for free VTU snapshots."""
    paths = sorted(
        [p for p in out_dir.glob("react___*.vtu")
         if not p.name.endswith("_C.vtu") and pod.vtu_step(p) is not None],
        key=lambda p: pod.vtu_step(p),
    )
    data = []
    for p in paths:
        step = pod.vtu_step(p)
        snap = pod.parse_vtu(p)
        fx = pod.front_position(snap, t_mid)
        if fx is not None and step is not None:
            data.append((step, float(snap["time"]), fx))
    return data


def free_phase_end(x_front: np.ndarray, tol: float = 2e-7):
    """Return index of last point before the front pins at the boundary."""
    dx = np.diff(x_front)
    for i in range(len(dx)):
        if abs(dx[i]) < tol:
            return i + 1
    return len(x_front)


def fit_sliding_window(t_code, x_front, free_end, steps_all, n_pts=5, t_max=0.6):
    """Slide an n_pts window through points with t_code < t_max, pick the
    best R² window from the later half of the eligible range.

    Returns (marker_speed_m_s, r_squared, t_mid, x_mid, mid_step).
    """
    eligible = t_code[:free_end] < t_max
    if eligible.sum() < n_pts:
        eligible = np.ones(free_end, dtype=bool)

    t_elig = t_code[:free_end][eligible]
    x_elig = x_front[:free_end][eligible]
    s_elig = steps_all[:free_end][eligible]
    n_total = len(t_elig)
    if n_total < n_pts:
        n_pts = n_total
        start_best = 0
    else:
        n_windows = n_total - n_pts + 1
        later_start = n_windows // 2
        best_r2 = -np.inf
        start_best = later_start
        for s in range(later_start, n_windows):
            t_w = t_elig[s:s + n_pts]
            x_w = x_elig[s:s + n_pts]
            coeffs = np.polyfit(t_w, x_w, 1)
            x_pred = np.polyval(coeffs, t_w)
            ss_res = np.sum((x_w - x_pred) ** 2)
            ss_tot = np.sum((x_w - np.mean(x_w)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
            if r2 > best_r2:
                best_r2 = r2
                start_best = s

    t_mid = t_elig[start_best:start_best + n_pts]
    x_mid = x_elig[start_best:start_best + n_pts]

    coeffs = np.polyfit(t_mid, x_mid, 1)
    x_pred = np.polyval(coeffs, t_mid)
    ss_res = np.sum((x_mid - x_pred) ** 2)
    ss_tot = np.sum((x_mid - np.mean(x_mid)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    marker = coeffs[0] * U0
    mid_step = int(s_elig[start_best + n_pts // 2])
    return marker, r2, t_mid, x_mid, mid_step


def unburned_velocity(pod, out_dir, step, t_mid):
    """Average u in unburned region ahead of flame at given VTU step."""
    p = out_dir / f"react___{step}.vtu"
    if not p.exists():
        return float("nan")
    snap = pod.parse_vtu(p)
    fx = pod.front_position(snap, t_mid)
    x_arr = np.asarray(snap["x"], float).reshape(-1)
    velo = np.asarray(snap["cell_data"]["Velo"], float).reshape((-1, 3))[:, 0]
    order = np.argsort(x_arr)
    xs, us = x_arr[order], velo[order] * U0
    mask = (xs > fx + 0.002) & (xs < 0.018)
    return float(np.mean(us[mask])) if mask.any() else float("nan")


# ---- Plotting ----

_COLORS = ["tab:blue", "tab:orange", "tab:cyan", "tab:red",
           "tab:green", "tab:purple", "tab:brown", "tab:pink"]
_LINESTYLES = ["-", "--", "-.", ":"]


def plot_all(runs_data, out_path: Path, su_ref: float, dpi: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, rd in enumerate(runs_data):
        color = _COLORS[i % len(_COLORS)]
        ls = _LINESTYLES[i % len(_LINESTYLES)]

        t_all = rd["t_all"]
        x_all = rd["x_all"]
        t_phys = t_all / U0 * 1e3
        t_mid = rd["t_mid"]
        x_mid = rd["x_mid"]
        free_end = rd["free_end"]
        label = rd["label"]
        Su = rd["Su"]

        ax1.plot(t_all[:free_end], x_all[:free_end] * 1e3, "o-", ms=2.5, lw=1.2,
                 color=color, label=label)
        ax1.plot(t_mid, x_mid * 1e3, "s", ms=6, color=color, mec="k", mew=0.8)
        if free_end < len(t_all):
            ax1.plot(t_all[free_end - 1:], x_all[free_end - 1:] * 1e3,
                     "x-", ms=3, lw=0.8, color=color, alpha=0.3)

        su_s = f"{Su:.2f} m/s ({Su / su_ref:.2f}×)" if np.isfinite(Su) else "n/a"
        ax2.plot(t_phys[:free_end], x_all[:free_end] * 1e3, "o-", ms=2.5, lw=1.2,
                 color=color, label=f"{label}: Su≈{su_s}")
        ax2.plot(t_mid / U0 * 1e3, x_mid * 1e3, "s",
                 ms=6, color=color, mec="k", mew=0.8)
        if free_end < len(t_all):
            ax2.plot(t_phys[free_end - 1:], x_all[free_end - 1:] * 1e3,
                     "x-", ms=3, lw=0.8, color=color, alpha=0.3)

    ax1.set_xlabel("t (code units)")
    ax1.set_ylabel("front x (mm)")
    ax1.set_title("Front Position — mid-range fit points marked (□)")
    ax1.legend(fontsize=7.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("t_phys (ms)")
    ax2.set_ylabel("front x (mm)")
    ax2.set_title(
        f"Front Position vs Physical Time (Cantera Su={su_ref:.2f} m/s)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---- Main ----

def main():
    args = parse_args()
    pod = _load_plot_digest()
    t_mid = args.t_mid or 0.5 * (CANTERA_T_FRESH + CANTERA_T_BURNED)
    labels = args.labels or [d.name for d in args.dirs]

    if len(labels) != len(args.dirs):
        labels = [d.name for d in args.dirs]

    runs_data = []

    for d, label in zip(args.dirs, labels):
        data = extract_front_data(pod, d, t_mid)
        if not data:
            print(f"WARNING: no front data in {d}", file=sys.stderr)
            continue

        t_code = np.array([r[1] for r in data])
        x_front = np.array([r[2] for r in data])
        steps = np.array([r[0] for r in data])

        free_end = free_phase_end(x_front)
        marker, r2, t_mid_vals, x_mid_vals, mid_step = fit_sliding_window(
            t_code, x_front, free_end, steps, args.fit_pts)

        u_ub = unburned_velocity(pod, d, mid_step, t_mid)
        Su = marker - u_ub if np.isfinite(u_ub) else float("nan")

        runs_data.append({
            "label": label, "t_all": t_code, "x_all": x_front,
            "free_end": free_end, "marker": marker, "r2": r2,
            "t_mid": t_mid_vals, "x_mid": x_mid_vals,
            "u_ub": u_ub, "Su": Su,
        })

    if not runs_data:
        print("ERROR: no valid data found", file=sys.stderr)
        sys.exit(1)

    plot_all(runs_data, args.out, args.su_ref, args.dpi)
    print(f"Saved {args.out}")
    print()

    su_ref = args.su_ref
    print(f'{"Label":<30s} {"R²":>8s} {"Marker":>7s} {"u_ub":>6s} {"Su":>7s} {"vsCantera":>9s}')
    print("-" * 73)
    for rd in runs_data:
        Su = rd["Su"]
        su_s = f"{Su:>7.3f}" if np.isfinite(Su) else "    NaN"
        r = f"{Su / su_ref:>8.3f}x" if np.isfinite(Su) else "     NaN"
        print(
            f'{rd["label"]:<30s} {rd["r2"]:>8.6f} {rd["marker"]:>7.3f} {rd["u_ub"]:>6.3f} {su_s} {r}')


if __name__ == "__main__":
    main()
