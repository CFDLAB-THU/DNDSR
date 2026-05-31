#!/usr/bin/env python3
"""Plot DNDSR output profiles/history and write a compact run digest.

Example:
    python scripts/plot_output_digest.py data/out/my_run

The script auto-detects the latest output prefix in the directory, reads the
matching ``*.vtu`` snapshots and ``*.log`` file, then writes plots and
``digest.json`` under ``<output-dir>/digest_<prefix>/``.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import argparse
import base64
import csv
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TypedDict

import matplotlib

matplotlib.use("Agg")


SPECIES_LABELS = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR"]
CONSERVATIVE_LABELS = [
    "rho",
    "rhoU",
    "rhoV",
    "rhoW",
    "rhoE",
    "Y_H2",
    "Y_H",
    "Y_O",
    "Y_O2",
    "Y_OH",
    "Y_H2O",
    "Y_HO2",
    "Y_H2O2",
    "Y_AR",
]


class VTUSnapshot(TypedDict):
    path: str
    time: float
    x: np.ndarray
    cell_data: dict[str, np.ndarray]
    point_data: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path,
                        help="Directory containing DNDSR .vtu/.log output")
    parser.add_argument(
        "--prefix", help="Run prefix, for example react__2026-...; default: auto-detect latest")
    parser.add_argument("--out-dir", type=Path,
                        help="Directory for digest outputs; default: <output_dir>/digest_<prefix>")
    parser.add_argument("--profile-count", type=int, default=12,
                        help="Maximum VTU snapshots to overlay in profile plots")
    parser.add_argument("--front-temperature", type=float,
                        help="Temperature threshold for front tracking")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting existing digest files")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def decode_binary_array(payload: str, dtype: np.dtype, header_type: str) -> np.ndarray:
    raw = base64.b64decode("".join(payload.split()))
    header_size = 8 if header_type == "UInt64" else 4
    header_fmt = "<Q" if header_size == 8 else "<I"
    nbytes = struct.unpack(header_fmt, raw[:header_size])[0]
    body = raw[header_size: header_size + nbytes]
    return np.frombuffer(body, dtype=dtype).copy()


def dtype_from_vtk(type_name: str) -> np.dtype:
    mapping = {
        "Float64": np.dtype("<f8"),
        "Float32": np.dtype("<f4"),
        "Int64": np.dtype("<i8"),
        "Int32": np.dtype("<i4"),
        "UInt64": np.dtype("<u8"),
        "UInt32": np.dtype("<u4"),
        "UInt8": np.dtype("u1"),
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported VTK DataArray type: {type_name}")
    return mapping[type_name]


def attr_value(attrs: str, name: str, default: str | None = None) -> str | None:
    match = re.search(rf'\b{name}="([^"]*)"', attrs)
    return match.group(1) if match else default


def parse_data_arrays(block: str, header_type: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    pattern = re.compile(r"<DataArray\s+([^>]*)>\s*(.*?)\s*</DataArray>", re.S)
    unnamed_index = 0
    for match in pattern.finditer(block):
        attrs, payload = match.groups()
        name = attr_value(attrs, "Name") or attr_value(
            attrs, "name") or f"__unnamed_{unnamed_index}"
        unnamed_index += 1
        vtk_type = attr_value(attrs, "type", "Float64") or "Float64"
        fmt = attr_value(attrs, "format", "ascii")
        ncomp = int(attr_value(attrs, "NumberOfComponents", "1") or "1")
        if fmt == "binary":
            data = decode_binary_array(
                payload, dtype_from_vtk(vtk_type), header_type)
        elif fmt == "ascii":
            data = np.fromstring(
                payload, sep=" ", dtype=dtype_from_vtk(vtk_type))
        else:
            continue
        if ncomp > 1 and data.size % ncomp == 0:
            data = data.reshape((-1, ncomp))
        arrays[name] = data
    return arrays


def block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.S)
    return match.group(1) if match else ""


def parse_vtu(path: Path) -> VTUSnapshot:
    text = read_text(path)
    header_type = attr_value(text[:300], "header_type", "UInt32") or "UInt32"
    time_match = re.search(r'name="TIME"[^>]*>\s*([0-9.eE+-]+)', text)
    time = float(time_match.group(1)) if time_match else math.nan

    points_arrays = parse_data_arrays(block(text, "Points"), header_type)
    points = next(iter(points_arrays.values())
                  ) if points_arrays else np.empty((0, 3))
    cells = parse_data_arrays(block(text, "Cells"), header_type)
    cell_data = parse_data_arrays(block(text, "CellData"), header_type)
    point_data = parse_data_arrays(block(text, "PointData"), header_type)

    centers = cell_centers_x(points, cells, next(
        iter(cell_data.values())).size if cell_data else 0)
    return {
        "path": str(path),
        "time": time,
        "x": centers,
        "cell_data": cell_data,
        "point_data": point_data,
    }


def cell_centers_x(points: np.ndarray, cells: dict[str, np.ndarray], fallback_size: int) -> np.ndarray:
    if points.size == 0 or "connectivity" not in cells or "offsets" not in cells:
        return np.arange(fallback_size, dtype=float)
    points = np.asarray(points).reshape((-1, 3))
    conn = np.asarray(cells["connectivity"], dtype=np.int64)
    offsets = np.asarray(cells["offsets"], dtype=np.int64)
    centers = np.empty(offsets.size, dtype=float)
    start = 0
    for i, end in enumerate(offsets):
        ids = conn[start:end]
        centers[i] = float(np.mean(points[ids, 0])) if ids.size else float(i)
        start = int(end)
    return centers


def vtu_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.vtu$", path.name)
    return int(match.group(1)) if match else None


def prefix_from_vtu(path: Path) -> str | None:
    if path.name.endswith("_C.vtu"):
        return None
    match = re.match(r"(.+)_\d+\.vtu$", path.name)
    return match.group(1) if match else None


def choose_prefix(output_dir: Path, requested: str | None) -> str:
    if requested:
        return requested
    logs = sorted(output_dir.glob("*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        prefix = log.name[:-4]
        if any(output_dir.glob(f"{prefix}_*.vtu")):
            return prefix
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in output_dir.glob("*.vtu"):
        prefix = prefix_from_vtu(path)
        if prefix:
            groups[prefix].append(path)
    if not groups:
        raise FileNotFoundError(f"No VTU snapshots found in {output_dir}")
    return max(groups, key=lambda key: (len(groups[key]), max(p.stat().st_mtime for p in groups[key])))


def select_snapshots(output_dir: Path, prefix: str, max_count: int) -> list[Path]:
    paths = [p for p in output_dir.glob(
        f"{prefix}_*.vtu") if not p.name.endswith("_C.vtu")]
    paths = sorted(paths, key=lambda p: (
        vtu_step(p) is None, vtu_step(p) or -1, p.name))
    if len(paths) <= max_count:
        return paths
    keep = sorted(
        set(np.linspace(0, len(paths) - 1, max_count, dtype=int).tolist()))
    return [paths[i] for i in keep]


def load_log_rows(log_path: Path) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = [dict(row) for row in reader]
        header = list(reader.fieldnames or [])
    finals = [r for r in rows if r.get(
        "iStep") == "-1" and r.get("iter") == "-1"]
    if finals:
        return header, rows, finals
    by_step: dict[str, dict[str, str]] = {}
    for row in rows:
        by_step[row.get("step", "")] = row
    return header, rows, [by_step[k] for k in sorted(by_step, key=lambda x: int(float(x or 0)))]


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row[name]))
        except (KeyError, ValueError):
            vals.append(math.nan)
    return np.asarray(vals, dtype=float)


def finite_minmax(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def array_digest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float | None]]:
    return {name: finite_minmax(np.asarray(values, dtype=float).reshape(-1)) for name, values in sorted(arrays.items())}


def plot_profiles(snapshots: list[VTUSnapshot], out_dir: Path) -> list[str]:
    outputs: list[str] = []
    if not snapshots:
        return outputs
    thermo_names = ["T", "P", "R", "M", "RE"]
    species_names = [f"V{i}" for i in range(1, 10)]
    diagnostic_names = ["RHSr", "ifUseLimiter"]
    datasets = [
        ("profiles_thermo.png", thermo_names, "Thermodynamic Profiles"),
        ("profiles_species.png", species_names, "Species Profiles"),
        ("profiles_diagnostics.png", diagnostic_names, "Diagnostic Profiles"),
    ]
    for filename, names, title in datasets:
        names = [name for name in names if name in snapshots[-1]["cell_data"]]
        if not names:
            continue
        ncols = 2
        nrows = math.ceil(len(names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(
            13, 3.2 * nrows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            for snap in snapshots:
                x = np.asarray(snap["x"], dtype=float)
                y = np.asarray(snap["cell_data"][name],
                               dtype=float).reshape(-1)
                order = np.argsort(x)
                label = f"t={snap['time']:.4g}"
                ax.plot(x[order], y[order], lw=1.0, label=label)
            ylabel = name
            if name.startswith("V") and name[1:].isdigit():
                idx = int(name[1:]) - 1
                if 0 <= idx < len(SPECIES_LABELS):
                    ylabel = f"Y_{SPECIES_LABELS[idx]}"
            ax.set_title(ylabel)
            ax.set_xlabel("x")
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncols=min(6, len(labels)), fontsize=8)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        outputs.append(str(out_path))
    return outputs


def plot_history(header: list[str], finals: list[dict[str, str]], out_dir: Path) -> list[str]:
    if not finals:
        return []
    t = numeric_column(finals, "tSimu")
    step = numeric_column(finals, "step")
    x = t if np.isfinite(t).any() else step
    xlabel = "tSimu" if np.isfinite(t).any() else "step"
    outputs: list[str] = []

    res_cols = [name for name in header if re.fullmatch(r"res\d+", name)]
    if res_cols:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name in res_cols:
            y = np.abs(numeric_column(finals, name))
            if np.isfinite(y).any():
                ax.semilogy(x, y, lw=0.8, label=name)
        ax.set_title("Residual History")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("abs residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncols=4, fontsize=7)
        fig.tight_layout()
        out_path = out_dir / "history_residuals.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        outputs.append(str(out_path))

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for name in ["curDtImplicit", "curDtMin", "CFLNow"]:
        if name in header:
            axes[0].plot(x, numeric_column(finals, name), label=name)
    axes[0].set_title("Timestep And CFL")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for idx, label in enumerate(CONSERVATIVE_LABELS):
        lo, hi = f"uMin{idx}", f"uMax{idx}"
        if lo in header and hi in header:
            axes[1].plot(x, numeric_column(finals, lo),
                         lw=0.7, label=f"{label} min")
            axes[1].plot(x, numeric_column(finals, hi),
                         lw=0.7, ls="--", label=f"{label} max")
    axes[1].set_title("State Min/Max History")
    axes[1].legend(ncols=3, fontsize=6)
    axes[1].grid(True, alpha=0.3)

    for name in ["nLimInc", "alphaMinInc", "nLimBeta", "minBeta", "nLimAlpha", "minAlpha"]:
        if name in header:
            axes[2].plot(x, numeric_column(finals, name), label=name)
    axes[2].set_title("Limiter History")
    axes[2].set_xlabel(xlabel)
    axes[2].legend(ncols=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "history_state_limiters.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    outputs.append(str(out_path))
    return outputs


def front_position(snapshot: VTUSnapshot, threshold: float | None) -> float | None:
    if threshold is None or "T" not in snapshot["cell_data"]:
        return None
    x = np.asarray(snapshot["x"], dtype=float)
    t = np.asarray(snapshot["cell_data"]["T"], dtype=float).reshape(-1)
    order = np.argsort(x)
    x, t = x[order], t[order]
    diff = t - threshold
    crossings = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if crossings.size == 0:
        return None
    i = int(crossings[0])
    if t[i + 1] == t[i]:
        return float(x[i])
    frac = (threshold - t[i]) / (t[i + 1] - t[i])
    return float(x[i] + frac * (x[i + 1] - x[i]))


def write_digest(
    output_dir: Path,
    prefix: str,
    log_path: Path | None,
    all_rows: list[dict[str, str]],
    finals: list[dict[str, str]],
    snapshots: list[VTUSnapshot],
    selected_paths: list[Path],
    front_temperature: float | None,
    plot_paths: Iterable[str],
    out_dir: Path,
) -> Path:
    final_snapshot = snapshots[-1] if snapshots else None
    final_log = finals[-1] if finals else None
    digest: dict[str, object] = {
        "output_dir": str(output_dir),
        "prefix": prefix,
        "log_file": str(log_path) if log_path else None,
        "log_rows": len(all_rows),
        "step_final_rows": len(finals),
        "vtu_snapshots_used": [str(p) for p in selected_paths],
        "plots": list(plot_paths),
    }
    if final_log:
        digest["final_log"] = {
            "step": int(float(final_log.get("step", "nan"))),
            "tSimu": float(final_log.get("tSimu", "nan")),
            "CFLNow": float(final_log.get("CFLNow", "nan")),
            "curDtImplicit": float(final_log.get("curDtImplicit", "nan")),
            "residual_max_abs": float(
                np.nanmax([abs(float(v)) for k, v in final_log.items()
                          if re.fullmatch(r"res\d+", k)])
            ),
        }
    if final_snapshot:
        digest["final_snapshot"] = {
            "time": float(final_snapshot["time"]),
            "cell_data_minmax": array_digest(final_snapshot["cell_data"]),
            "front_temperature": front_temperature,
            "front_x": front_position(final_snapshot, front_temperature),
        }
    out_path = out_dir / "digest.json"
    out_path.write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n")
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    prefix = choose_prefix(output_dir, args.prefix)
    out_dir = (args.out_dir or output_dir / f"digest_{prefix}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{out_dir} is not empty; pass --force to overwrite digest files")

    selected_paths = select_snapshots(
        output_dir, prefix, max(1, args.profile_count))
    snapshots = [parse_vtu(path) for path in selected_paths]

    log_path = output_dir / f"{prefix}.log"
    header: list[str] = []
    all_rows: list[dict[str, str]] = []
    finals: list[dict[str, str]] = []
    if log_path.is_file():
        header, all_rows, finals = load_log_rows(log_path)
    else:
        log_path = None

    plot_paths = []
    plot_paths.extend(plot_profiles(snapshots, out_dir))
    if header and finals:
        plot_paths.extend(plot_history(header, finals, out_dir))
    digest_path = write_digest(
        output_dir=output_dir,
        prefix=prefix,
        log_path=log_path,
        all_rows=all_rows,
        finals=finals,
        snapshots=snapshots,
        selected_paths=selected_paths,
        front_temperature=args.front_temperature,
        plot_paths=plot_paths,
        out_dir=out_dir,
    )

    print(f"prefix={prefix}")
    print(f"digest_dir={out_dir}")
    print(f"snapshots={len(selected_paths)}")
    print(f"log_rows={len(all_rows)} step_finals={len(finals)}")
    print(f"digest={digest_path}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
