#!/usr/bin/env python3
"""Detonation speed analysis from 2-D VTKHDF output via y-averaged 1-D profiles.

Usage:
    python analyze_detonation_2d.py \\
        --series data/.../react_-T0-fine_.vtkhdf.series --label "Strang dt=2e-6" \\
        --series data/.../out-T0_.vtkhdf.series --label "CS" \\
        --out-dir workspace/detonation2d/figs
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

matplotlib.use("Agg")

# DNDSR non-dimensionalisation
U0 = 379.0       # velocity scale [m/s]
P0 = U0 * U0     # pressure scale = rho0 * U0^2


# ---------------------------------------------------------------------------
# 2-D → 1-D profile extraction
# ---------------------------------------------------------------------------

def _extract_worker(args_tuple):
    """Multiprocessing worker: read one VTKHDF, return y-averaged 1D profile."""
    fpath, n_xbins = args_tuple
    import pyvista as pv_worker
    mesh = pv_worker.read(fpath)
    return extract_1d_profile(mesh, n_xbins)


def extract_1d_profile(mesh, n_xbins: int = 0):
    """Y-average cell data and return 1-D profiles on x-binned centres.

    If n_xbins <= 0, auto-detect from mesh: count unique cell-centre
    x-coordinates (tolerance = 1e-12 × domain width).
    """
    cc = mesh.cell_centers().points
    x = cc[:, 0]

    if n_xbins <= 0:
        x_sorted = np.sort(x)
        dx = np.diff(x_sorted)
        n_xbins = int(
            np.sum(dx > 1e-12 * (mesh.bounds[1] - mesh.bounds[0]))) + 1
        n_xbins = max(100, n_xbins)

    xb = np.linspace(mesh.bounds[0], mesh.bounds[1], n_xbins + 1)
    xc = 0.5 * (xb[1:] + xb[:-1])
    dig = np.clip(np.digitize(x, xb) - 1, 0, n_xbins - 1)

    # Y1D columns (look up y-averages by name)
    def _yavg(name):
        data = mesh[name]
        if data.ndim == 2:
            data = data[:, 0]          # Velo → x-component
        vals = np.array([data[dig == i].mean() if np.any(dig == i) else np.nan
                         for i in range(n_xbins)])
        return vals

    T = _yavg("T")                                           # [K]
    P_bar = _yavg("P") * P0 / 1e5                            # [bar]
    u_mps = _yavg("Velo") * U0                               # [m/s]
    R = _yavg("R")                                           # code units

    # species mass fractions (anything starting with Y_)
    species = {}
    for key in mesh.cell_data.keys():
        if key.startswith("Y_"):
            species[key] = _yavg(key)

    valid = ~np.isnan(T)
    result = {"x": xc[valid], "T": T[valid], "P_bar": P_bar[valid],
              "u_mps": u_mps[valid], "R": R[valid], "species": {}}
    for k, v in species.items():
        result["species"][k] = v[valid]
    return result


# ---------------------------------------------------------------------------
# Shock tracking
# ---------------------------------------------------------------------------

def shock_front(profile: dict, p_thresh: float = 1.5):
    """Rightmost x where P_bar > p_thresh. Returns (x_meters, index)."""
    shocked = profile["P_bar"] > p_thresh
    if not shocked.any():
        return None, None
    idx = int(np.where(shocked)[0][-1])
    return float(profile["x"][idx]), idx


# ---------------------------------------------------------------------------
# Speed estimation (sliding-window linear fit)
# ---------------------------------------------------------------------------

def sliding_fit(t_code: np.ndarray, x_front: np.ndarray,
                n_pts: int = 5, t_max: float = 1e100):
    """Slide n_pts window through points where t_code < t_max,
    pick best R² from the later half.  Returns speed (m/s), R², window data."""
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


# ---------------------------------------------------------------------------
# Profile plots
# ---------------------------------------------------------------------------

def plot_profiles(snapshots, label: str, out_path: str,
                  n_snapshots: int = 8, dpi: int = 180):
    """3-panel (T, P, u) profiles at evenly-spaced snapshots."""
    if len(snapshots) < 2:
        return
    idx = np.linspace(0, len(snapshots) - 1,
                      min(n_snapshots, len(snapshots)), dtype=int)
    idx = np.unique(idx)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(idx)))

    for i, j in enumerate(idx):
        s = snapshots[j]
        lab = f"t={s['time']:.4e}"
        axes[0].plot(s["profile"]["x"] * 1e3, s["profile"]["T"],
                     lw=1.2, color=colors[i], label=lab)
        axes[1].plot(s["profile"]["x"] * 1e3, s["profile"]["P_bar"],
                     lw=1.2, color=colors[i])
        axes[2].plot(s["profile"]["x"] * 1e3, s["profile"]["u_mps"],
                     lw=1.2, color=colors[i])

    axes[0].set_ylabel("T (K)")
    axes[1].set_ylabel("P (bar)")
    axes[2].set_ylabel("u (m/s)")
    axes[0].set_title(f"{label} — Temperature")
    axes[1].set_title("Pressure")
    axes[2].set_title("Velocity")
    axes[0].legend(fontsize=7.5, ncols=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x (mm)")
    fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Profile plot: {out_path}")


# ---------------------------------------------------------------------------
# Per-series analysis
# ---------------------------------------------------------------------------

_COLORS = ["tab:blue", "tab:orange", "tab:cyan", "tab:red",
           "tab:green", "tab:purple", "tab:brown", "tab:pink"]


def analyze_series(series_path: str, label: str, args, color: str, out_dir: str):
    data_dir = os.path.dirname(os.path.abspath(series_path))

    with open(series_path) as f:
        series = json.load(f)
    files = sorted(
        [(e["name"], e["time"]) for e in series["files"]],
        key=lambda x: x[1],
    )

    # Build task list (only existing files)
    tasks = []
    n_missing = 0
    for fname, tval in files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            tasks.append((fpath, tval, fname))
        else:
            n_missing += 1
    if n_missing:
        print(f"  SKIP {n_missing} missing files (of {len(files)} in series)")

    n_jobs = args.n_jobs
    if n_jobs > 1:
        # Use spawn context to avoid fork+VTK issues
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        from concurrent.futures import ProcessPoolExecutor as PPE, as_completed

        print(
            f"  Loading {len(tasks)} frames with {n_jobs} workers (spawn) ...")
        futures_map = {}
        with PPE(max_workers=n_jobs, mp_context=ctx) as ex:
            for fpath, tval, fname in tasks:
                fut = ex.submit(_extract_worker, (fpath, args.n_xbins))
                futures_map[fut] = (tval, fname)

            snapshots = []
            for i, fut in enumerate(as_completed(futures_map), 1):
                tval, fname = futures_map[fut]
                profile = fut.result()
                snapshots.append(
                    {"time": tval, "fname": fname, "profile": profile})
                if i % 1 == 0:
                    print(f"    [{i}/{len(tasks)}] done, latest: {fname}")
    else:
        print(f"  Loading {len(tasks)} frames serial ...")
        snapshots = []
        for i, (fpath, tval, fname) in enumerate(tasks, 1):
            profile = extract_1d_profile(pv.read(fpath), args.n_xbins)
            snapshots.append(
                {"time": tval, "fname": fname, "profile": profile})
            print(f"    [{i}/{len(tasks)}] {fname} (t={tval:.4e})")

    # Restore time order
    snapshots.sort(key=lambda s: s["time"])

    if not snapshots:
        print(f"{label}: no snapshots found")
        return None, None, None, None

    # Track shock front
    times, x_m = [], []
    for s in snapshots:
        fx, _ = shock_front(s["profile"], args.p_threshold)
        if fx is not None:
            times.append(s["time"])
            x_m.append(fx)

    t_code = np.array(times)
    x_front = np.array(x_m)

    # Speed estimation
    speed, r2, t_mid, x_mid = sliding_fit(
        t_code, x_front, args.fit_pts, args.t_max)

    print(f"{label}: {len(t_code)} shock pts, "
          f"speed={speed:.1f} m/s, R²={r2:.6f}")
    if len(t_code) > 0:
        print(f"  start: x={x_front[0]*1e3:.3f} mm  t={t_code[0]:.6e}")
        print(f"  end:   x={x_front[-1]*1e3:.3f} mm  t={t_code[-1]:.6e}")

    # Profile plot
    slug = label.replace(" ", "_").replace("/", "-")[:40]
    prof_path = os.path.join(out_dir, f"profiles_{slug}.png")
    plot_profiles(snapshots, f"{label} — profiles",
                  prof_path, dpi=args.dpi)

    return t_code, x_front, (t_mid, x_mid), speed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", action="append", default=[],
                   help="Path to .vtkhdf.series file (repeatable)")
    p.add_argument("--labels", nargs="*",
                   help="Labels for each series (same order as --series)")
    p.add_argument("--out-dir", default="workspace/detonation2d/figs",
                   help="Output directory for plots")
    p.add_argument("--p-threshold", type=float, default=1.5,
                   help="Pressure ratio for shock front (default 1.5 x ambient)")
    p.add_argument("--fit-pts", type=int, default=5,
                   help="Sliding-window points (default 5)")
    p.add_argument("--t-max", type=float, default=1e100,
                   help="Max code time for fitting window")
    p.add_argument("--n-xbins", type=int, default=0,
                   help="Number of x-bins for y-averaging (0 = auto-detect from mesh)")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel workers for profile extraction (default 4)")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    if not args.series:
        p.error("At least one --series is required")

    labels = args.labels or [os.path.basename(os.path.dirname(s))
                             for s in args.series]
    if len(labels) != len(args.series):
        labels = [os.path.basename(os.path.dirname(s))[-40:]
                  for s in args.series]

    os.makedirs(args.out_dir, exist_ok=True)

    # Shock position vs time plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for i, (spath, label) in enumerate(zip(args.series, labels)):
        color = _COLORS[i % len(_COLORS)]
        t_code, x_front, fit_data, speed = analyze_series(
            spath, label, args, color, args.out_dir)

        if t_code is None or len(t_code) == 0:
            continue

        t_phys = t_code / U0 * 1e6    # μs

        # Code-units plot
        ax1.plot(t_code, x_front * 1e3, "o-", ms=2, lw=1,
                 color=color, label=label)
        if fit_data is not None:
            t_mid, x_mid = fit_data
            ax1.plot(t_mid, x_mid * 1e3, "s", ms=6,
                     color=color, mec="k", mew=0.8)

        # Physical plot
        label2 = f"{label}: {speed:.0f} m/s" if np.isfinite(speed) else label
        ax2.plot(t_phys, x_front * 1e3, "o-", ms=2,
                 lw=1, color=color, label=label2)
        if fit_data is not None:
            t_mid, x_mid = fit_data
            ax2.plot(t_mid / U0 * 1e6, x_mid * 1e3, "s", ms=6,
                     color=color, mec="k", mew=0.8)

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
    out_path = os.path.join(args.out_dir, "detonation_speed.png")
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
