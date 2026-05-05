#!/usr/bin/env python3
"""Plot TGV weak-scaling benchmark (CSV → SVG).

Input:  tgv_weak_2025-03.csv
Output: tgv_weak_scaling.svg (to docs/presentations/res/)

Layout: two-panel — left = total throughput (kCI/s), right = per-core efficiency.
Only BSSCA series is plotted.  CSV rows with empty first column inherit the
label from the last labelled row.

Efficiency = T_base / T  (weak-scaling: fixed work per rank, ideal time = T_base).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "tgv_weak_2025-03.csv"
OUT_DIR = HERE.parent.parent / "presentations" / "res"
OUT_SVG = OUT_DIR / "tgv_weak_scaling.svg"

# ---------------------------------------------------------------------------
# Parse CSV — empty first column inherits the previous label.
# Raw columns: (empty), ncpu, NX, Time, Niter, ...
# Compute throughput: kCI/s = NX³ cells × Niter / (time × 1000)
# ---------------------------------------------------------------------------
pts: list[dict] = []
cur_label = ""
with CSV_PATH.open(newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        label = row[0].strip()
        if label and not label.startswith(","):
            cur_label = label
        if cur_label != "BSSCA":
            continue
        try:
            ncpu = int(row[1])
            nx = int(row[2])
            time_val = float(row[3])
            niter = int(row[4])
        except (ValueError, IndexError):
            continue
        ncell = nx ** 3
        kcis = ncell * niter / (time_val * 1000)            # total throughput
        kcis_per_core = kcis / ncpu                         # per-core throughput
        pts.append({"ncpu": ncpu, "nx": nx, "kcis": kcis,
                   "kcis_per_core": kcis_per_core})

pts.sort(key=lambda p: p["ncpu"])
ncpus = np.array([p["ncpu"] for p in pts])
kcis = np.array([p["kcis"] for p in pts])
kcis_per_core = np.array([p["kcis_per_core"] for p in pts])
base_kcis_per_core = kcis_per_core[0]
eff_pct = kcis_per_core / base_kcis_per_core * 100   # weak-scaling efficiency

COLOR = "#660874"
# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_tput, ax_eff) = plt.subplots(
    1, 2, figsize=(12, 5.2), constrained_layout=True)
fig.patch.set_facecolor("white")

# --- Throughput panel (log–log) -------------------------------------------
ax_tput.loglog(ncpus, kcis, "s-", color=COLOR, markersize=8, linewidth=2.0,
               markerfacecolor="white", markeredgewidth=1.5, label="BSSCA")

ref_ncpu = np.array([ncpus[0], ncpus[-1]])
ref_tput = np.array([kcis[0], kcis[0] * ncpus[-1] / ncpus[0]])
ax_tput.loglog(ref_ncpu, ref_tput, "k--",
               linewidth=0.8, alpha=0.35, label="ideal")

ax_tput.set_xlabel("MPI ranks", fontsize=11)
ax_tput.set_ylabel("kCI/s  (total throughput)", fontsize=11)
ax_tput.set_title("Weak scaling — total throughput",
                  fontsize=12, fontweight="bold")
ax_tput.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax_tput.grid(True, which="both", alpha=0.2)

# --- Efficiency panel (linear) --------------------------------------------
ax_eff.plot(ncpus, eff_pct, "s-", color=COLOR, markersize=8, linewidth=2.0,
            markerfacecolor="white", markeredgewidth=1.5)
ax_eff.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
for x, y in zip(ncpus, eff_pct):
    ax_eff.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#444444")
ax_eff.set_xlabel("MPI ranks", fontsize=11)
ax_eff.set_ylabel("Weak-scaling efficiency  (%)", fontsize=11)
ax_eff.set_title("Weak scaling — per-core efficiency",
                 fontsize=12, fontweight="bold")
ax_eff.set_ylim(0, 115)
ax_eff.grid(True, alpha=0.2)

# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_SVG), dpi=150, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT_SVG}")
