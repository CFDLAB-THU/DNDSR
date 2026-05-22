#!/usr/bin/env python3
import base64
import csv
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REF_CSV = ROOT / "react_test_dense_50us_cantera_phys_history.csv"
SOLVER_CSV = ROOT / "react_test_dense_50us_eulerEX_history.csv"
OUT_PATH = ROOT / "react_test_dense_50us_compare.png"
OUT_DIR = ROOT / "eulerEX_dense_50us"
U0 = 379.0
P0 = U0 * U0


def decode_vtk_binary_doubles(text):
    raw = base64.b64decode("".join(text.split()))
    nbytes = struct.unpack("<Q", raw[:8])[0]
    return struct.unpack("<" + "d" * (nbytes // 8), raw[8: 8 + nbytes])


def extract_array(cell_block, name):
    match = re.search(
        rf'<DataArray[^>]*(?:Name|name)="{re.escape(name)}"[^>]*format="binary"[^>]*>\s*(.*?)\s*</DataArray>',
        cell_block,
        re.S,
    )
    if match is None:
        raise KeyError(name)
    vals = decode_vtk_binary_doubles(match.group(1))
    return sum(vals) / len(vals)


def parse_vtu(path):
    text = path.read_text(errors="ignore")
    time_match = re.search(r'name="TIME"[^>]*>\s*([0-9.eE+-]+)', text)
    if time_match is None:
        raise ValueError(f"missing TIME in {path}")
    cell_match = re.search(r"<CellData.*?</CellData>", text, re.S)
    if cell_match is None:
        raise ValueError(f"missing CellData in {path}")
    time = float(time_match.group(1))
    cell = cell_match.group(0)
    row = {
        "t_code": time,
        "t_phys": time / U0,
        "T_eulerEX": extract_array(cell, "T"),
        "p_eulerEX": extract_array(cell, "P") * P0,
    }
    species = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR"]
    for i, sp in enumerate(species, start=1):
        row[f"Y_{sp}_eulerEX"] = extract_array(cell, f"V{i}")
    row["Y_N2_eulerEX"] = 1.0 - sum(row[f"Y_{sp}_eulerEX"] for sp in species)
    return row


def latest_prefix():
    logs = sorted(OUT_DIR.glob("react__*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError(f"no logs under {OUT_DIR}")
    return logs[0].name[:-4]


def load_reference():
    with REF_CSV.open(newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def ignition_time(rows, key, threshold):
    for a, b in zip(rows, rows[1:]):
        if a[key] <= threshold <= b[key]:
            frac = (threshold - a[key]) / max(b[key] - a[key], 1e-300)
            return a["t_phys"] + frac * (b["t_phys"] - a["t_phys"])
    return rows[-1]["t_phys"]


prefix = latest_prefix()
vtu_files = sorted(
    [p for p in OUT_DIR.glob(prefix + "_*.vtu")
     if re.search(r"_(?:0+|\d+)\.vtu$", p.name)],
    key=lambda p: int(re.search(r"_(\d+)\.vtu$", p.name).group(1)),
)
solver_rows = [parse_vtu(p) for p in vtu_files]

with SOLVER_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(solver_rows[0].keys()))
    writer.writeheader()
    writer.writerows(solver_rows)

ref = load_reference()
ref = [r for r in ref if r["t_phys"] <= 50e-6]
solver_rows = [r for r in solver_rows if r["t_phys"] <= 50e-6]
t_ref_us = [r["t_phys"] * 1e6 for r in ref]
t_sol_us = [r["t_phys"] * 1e6 for r in solver_rows]

threshold = ref[0]["T_cantera"] + 0.5 * \
    (ref[-1]["T_cantera"] - ref[0]["T_cantera"])
tau_cantera = ignition_time(ref, "T_cantera", threshold)
tau_phys = ignition_time(ref, "T_dnds", threshold)
tau_euler = ignition_time(solver_rows, "T_eulerEX", threshold)

fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

ax = axes[0, 0]
ax.plot(t_ref_us, [r["T_cantera"]
        for r in ref], label="Cantera ReactorNet", lw=2.2, ls="--")
ax.plot(t_ref_us, [r["T_dnds"]
        for r in ref], label="Physics/0D reproduction", lw=2.0)
ax.plot(t_sol_us, [r["T_eulerEX"] for r in solver_rows],
        label="eulerEX", lw=1.8, marker=".", ms=3)
for tau, color, label in [(tau_cantera, "tab:blue", "Cantera tau"), (tau_phys, "tab:orange", "0D tau"), (tau_euler, "tab:green", "eulerEX tau")]:
    ax.axvline(tau * 1e6, color=color, alpha=0.25, lw=1.5)
ax.set_title("Temperature, dense 0-50 us")
ax.set_xlabel("physical time [us]")
ax.set_ylabel("T [K]")
ax.set_xlim(0, 50)
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(t_ref_us, [r["p_cantera"] / 1e5 for r in ref],
        label="Cantera ReactorNet", lw=2.2, ls="--")
ax.plot(t_ref_us, [r["p_dnds"] / 1e5 for r in ref],
        label="Physics/0D reproduction", lw=2.0)
ax.plot(t_sol_us, [r["p_eulerEX"] / 1e5 for r in solver_rows],
        label="eulerEX", lw=1.8, marker=".", ms=3)
ax.set_title("Pressure, dense 0-50 us")
ax.set_xlabel("physical time [us]")
ax.set_ylabel("p [bar]")
ax.set_xlim(0, 50)
ax.grid(True, alpha=0.3)
ax.legend()

species = ["H2", "O2", "H2O", "OH"]
colors = {"H2": "tab:blue", "O2": "tab:orange",
          "H2O": "tab:green", "OH": "tab:red"}
ax = axes[1, 0]
for sp in species:
    ax.plot(t_ref_us, [r[f"Y_{sp}_cantera"] for r in ref],
            color=colors[sp], lw=2.0, ls="--", label=f"{sp} Cantera")
    ax.plot(t_ref_us, [r[f"Y_{sp}_dnds"] for r in ref],
            color=colors[sp], lw=1.6, label=f"{sp} 0D")
    ax.plot(t_sol_us, [r[f"Y_{sp}_eulerEX"] for r in solver_rows],
            color=colors[sp], lw=1.2, marker=".", ms=2.5, label=f"{sp} eulerEX")
ax.set_title("Species, dense 0-50 us")
ax.set_xlabel("physical time [us]")
ax.set_ylabel("mass fraction")
ax.set_xlim(0, 50)
ax.grid(True, alpha=0.3)
ax.legend(ncols=3, fontsize=7)

ax = axes[1, 1]
for sp in species:
    ax.semilogy(t_ref_us, [abs(r[f"Y_{sp}_dnds"] - r[f"Y_{sp}_cantera"])
                for r in ref], color=colors[sp], lw=2, label=f"{sp} 0D-Cantera")
ax.set_title("0D reproduction vs Cantera species error")
ax.set_xlabel("physical time [us]")
ax.set_ylabel("absolute mass-fraction error")
ax.set_xlim(0, 50)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle(
    f"react_test dense comparison: tau Cantera={tau_cantera*1e6:.3f} us, 0D={tau_phys*1e6:.3f} us, eulerEX={tau_euler*1e6:.3f} us",
    fontsize=13,
)
fig.savefig(OUT_PATH, dpi=200)
print(f"solver_csv={SOLVER_CSV}")
print(f"plot={OUT_PATH}")
print(f"samples ref={len(ref)} eulerEX={len(solver_rows)}")
print(
    f"ignition_us cantera={tau_cantera*1e6:.9f} phys0d={tau_phys*1e6:.9f} eulerEX={tau_euler*1e6:.9f}")
