#!/usr/bin/env python3
"""Generate exprtk expressions to inject a ZND detonation profile as initial conditions.

Reads a ZND profile from an .npz file (produced by estimate_induction.py) and
generates exprtk expression strings suitable for the ``exprtkInitializers``
section of DNDSR solver JSON configs.

The generated expression uses piecewise linear interpolation over adaptively
sampled data vectors.  All parameters (Ly, shock position, perturbation
amplitudes) are emitted as named ``var`` declarations at the top of the
expression, so they can be tweaked directly in the JSON config without
re-running the generator.

State convention: ``primTP_phy`` — [T, u, v, w, P, Y0, ..., Y_{ns-2}].
Velocities are in the lab frame (unreacted gas at rest).  The shock
propagates in the +x direction; ``dist = x_shock - x[0]`` is the distance
behind the shock (positive behind, negative ahead).

Usage:
    python generate_znd_exprtk.py \\
        --znd-profile data/out/znd_H2O2_6667Pa.npz \\
        --Ly 0.1 --shock-pert-amp 1e-3 --v-pert-amp 40 \\
        --output data/out/znd_exprtk_init.json

    # Then copy the "exprtkInitializers" array into your solver config.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def find_cj_distance(dist, P, tol=0.01):
    """Return the distance at which pressure first falls within *tol* of its final value."""
    P_final = P[-1]
    idx = np.argmax(np.abs(P - P_final) / P_final < tol)
    return dist[idx]


def adaptive_resample(dist_full, n_points, ind_len, L_cj):
    """Resample *dist_full* with three density tiers: dense near the induction
    zone (0–3 ind_len), moderate through the reaction zone (3–15 ind_len),
    and sparse in the equilibrium tail (15 ind_len–L_cj)."""
    n1 = max(n_points // 3, 8)
    n2 = max(n_points // 3, 8)
    n3 = max(n_points - n1 - n2, 6)

    b1 = min(3 * ind_len, L_cj * 0.3)
    b2 = min(15 * ind_len, L_cj * 0.8)

    x1 = np.linspace(0, b1, n1, endpoint=False)
    x2 = np.linspace(b1, b2, n2, endpoint=False)
    x3 = np.linspace(b2, L_cj, n3, endpoint=True)

    xnew = np.concatenate([x1, x2, x3])
    xnew = np.clip(xnew, dist_full[0], dist_full[-1])
    return np.unique(xnew)


def interp_profile(x_new, dist, arrays):
    """Linearly interpolate all arrays in *arrays* onto *x_new* sample points."""
    return {k: np.interp(x_new, dist, v) for k, v in arrays.items()}


def fmt_vec(name, vals, per_line=6):
    """Format *vals* as an exprtk inline vector declaration ``var name[N] := {...};``."""
    n = len(vals)
    parts = []
    parts.append(f"var {name}[{n}] := {{")
    for i in range(0, n, per_line):
        chunk = vals[i:i + per_line]
        s = ", ".join(f"{v:.8g}" for v in chunk)
        if i + per_line < n:
            s += ","
        parts.append("    " + s)
    parts.append("};")
    return parts


def generate_exprtk(npz_path, n_points=50, Ly=0.1, shock_pert_amp=1e-3,
                    v_pert_amp=40.0, x_shock_override=None, cj_tol=0.01,
                    pocket=None):
    """Build the exprtk initializer dict from a ZND profile .npz file.

    Returns a dict with keys ``"exprtkInitializers"`` (ready for JSON config
    injection) and ``"_info"`` (metadata for human inspection).

    The shock is placed at *x_shock_override* or 1.5 × L_CJ from x = 0.
    Perturbations (shock position cosine, transverse velocity sine) are
    parameterised as ``var`` declarations at the top of the expression so
    they can be edited in-place without re-running the generator.

    If *pocket* is a dict, a rectangular pocket of unreacted gas is added
    behind the shock.  Expected keys: ``offset`` (distance from shock to
    nearest edge [m]), ``size_x``, ``size_y`` [m], ``T`` [K], ``P`` [Pa],
    and optionally ``y_center`` [m] (defaults to Ly/2).  The pocket
    overrides T, P, and species but preserves the interpolated velocity.
    """
    d = np.load(npz_path, allow_pickle=True)

    dist = d["distance"]
    T = d["T"]
    P = d["P"]
    U_sf = d["U_loc"]
    species = d["species"]
    D = float(d["D"][0])
    T1 = float(d["T1"][0])
    P1 = float(d["P1"][0])
    ind_len = float(d["ind_len"][0])
    Y1 = d["Y1"]
    species_names = list(d["species_names"])
    n_species = len(species_names)

    u_lab = D - U_sf

    L_cj = find_cj_distance(dist, P, tol=cj_tol)
    x_shock = x_shock_override if x_shock_override else 1.5 * L_cj

    x_new = adaptive_resample(dist, n_points, ind_len, L_cj)
    n_pts = len(x_new)

    arrays = {"T": T, "P": P, "u": u_lab}
    for i in range(n_species):
        arrays[f"Y{i}"] = species[i] if species.shape[0] == n_species else species[:, i]

    sampled = interp_profile(x_new, dist, arrays)

    T_cj = sampled["T"][-1]
    P_cj = sampled["P"][-1]
    u_cj = sampled["u"][-1]
    Y_cj = [sampled[f"Y{i}"][-1] for i in range(n_species)]

    lines = []
    lines.append("inRegion := 1;")
    lines.append(f"var Ly := {Ly:.8g};")
    lines.append(f"var x_shock_0 := {x_shock:.8g};")
    lines.append(f"var A_shock := {shock_pert_amp:.8g};")
    lines.append(f"var A_v := {v_pert_amp:.8g};")
    lines.append(f"var ind_len := {ind_len:.8g};")
    lines.append("")
    lines.append("var pert_y := cos(2 * pi * x[1] / Ly);")
    lines.append("var pert_y_sin := sin(2 * pi * x[1] / Ly);")
    lines.append("var x_shock := x_shock_0 + A_shock * pert_y;")
    lines.append("var dist := x_shock - x[0];")
    lines.append("")

    lines.extend(fmt_vec("xd", x_new))
    lines.extend(fmt_vec("Td", sampled["T"]))
    lines.extend(fmt_vec("Pd", sampled["P"]))
    lines.extend(fmt_vec("ud", sampled["u"]))
    for i in range(n_species - 1):
        lines.extend(fmt_vec(f"Y{i}d", sampled[f"Y{i}"]))
    lines.append("")

    lines.append("if (dist < 0) {")
    lines.append(f"    UExprtk[0] := {T1:.8g};")
    lines.append(f"    UExprtk[1] := 0.0;")
    lines.append(f"    UExprtk[2] := 0.0;")
    lines.append(f"    UExprtk[3] := 0.0;")
    lines.append(f"    UExprtk[4] := {P1:.8g};")
    for i in range(n_species - 1):
        lines.append(f"    UExprtk[{5 + i}] := {Y1[i]:.8g};")
    lines.append("}")

    lines.append(f"else if (dist >= xd[{n_pts - 1}]) {{")
    lines.append(f"    UExprtk[0] := {T_cj:.8g};")
    lines.append(f"    UExprtk[1] := {u_cj:.8g};")
    lines.append(f"    UExprtk[2] := 0.0;")
    lines.append(f"    UExprtk[3] := 0.0;")
    lines.append(f"    UExprtk[4] := {P_cj:.8g};")
    for i in range(n_species - 1):
        lines.append(f"    UExprtk[{5 + i}] := {Y_cj[i]:.8g};")
    lines.append("}")

    lines.append("else {")
    lines.append("    var idx := 0;")
    lines.append("    var wt := 0.0;")
    lines.append("    for (var i := 0; i < xd[] - 1; i += 1) {")
    lines.append("        if (dist >= xd[i] and dist < xd[i + 1]) {")
    lines.append("            idx := i;")
    lines.append("            wt := (dist - xd[i]) / (xd[i + 1] - xd[i]);")
    lines.append("            break;")
    lines.append("        };")
    lines.append("    };")
    lines.append("    UExprtk[0] := Td[idx] + wt * (Td[idx + 1] - Td[idx]);")
    lines.append("    UExprtk[1] := ud[idx] + wt * (ud[idx + 1] - ud[idx]);")
    lines.append("    UExprtk[2] := 0.0;")
    lines.append("    UExprtk[3] := 0.0;")
    lines.append("    UExprtk[4] := Pd[idx] + wt * (Pd[idx + 1] - Pd[idx]);")
    for i in range(n_species - 1):
        lines.append(
            f"    UExprtk[{5 + i}] := Y{i}d[idx] + wt * (Y{i}d[idx + 1] - Y{i}d[idx]);")
    lines.append("};")
    lines.append("")

    lines.append("UExprtk[2] += A_v * pert_y_sin * exp(-dist / (2 * ind_len))"
                 " * clamp(0, dist / (0.1 * ind_len), 1);")
    lines.append("")

    if pocket is not None:
        p_off = pocket.get("offset", 0.003)
        p_sx = pocket.get("size_x", 0.01)
        p_sy = pocket.get("size_y", 0.014)
        p_T = pocket.get("T", 2100.0)
        p_P = pocket.get("P", 46667.0)
        p_yc = pocket.get("y_center")

        lines.append(f"var pocket_xR := x_shock_0 - {p_off:.8g};")
        lines.append(f"var pocket_xL := x_shock_0 - {p_off + p_sx:.8g};")
        if p_yc is not None:
            lines.append(f"var pocket_yC := {p_yc:.8g};")
        else:
            lines.append("var pocket_yC := Ly / 2;")
        lines.append(f"var pocket_yH := {p_sy / 2:.8g};")
        lines.append(
            "if (x[0] >= pocket_xL and x[0] <= pocket_xR"
            " and x[1] >= (pocket_yC - pocket_yH)"
            " and x[1] <= (pocket_yC + pocket_yH)) {")
        lines.append(f"    UExprtk[0] := {p_T:.8g};")
        lines.append(f"    UExprtk[4] := {p_P:.8g};")
        for i in range(n_species - 1):
            lines.append(f"    UExprtk[{5 + i}] := {Y1[i]:.8g};")
        lines.append("};")
        lines.append("")

    lines.append("0")

    result = {
        "exprtkInitializers": [
            {
                "exprs": lines,
                "stateType": "primTP_phy"
            }
        ],
        "_info": {
            "D_m_s": D,
            "cj_speed_m_s": float(d["cj_speed"][0]),
            "L_cj_mm": L_cj * 1e3,
            "x_shock_mm": x_shock * 1e3,
            "ind_len_mm": ind_len * 1e3,
            "n_interp_points": n_pts,
            "species_order": species_names,
            "Ly_m": Ly,
            "shock_pert_amp_mm": shock_pert_amp * 1e3,
            "v_pert_amp_m_s": v_pert_amp,
            "pocket": pocket,
        }
    }
    return result


def main():
    p = argparse.ArgumentParser(
        description="Generate exprtk expressions from ZND profile for solver IV injection")
    p.add_argument("--znd-profile", required=True,
                   help="Path to ZND .npz file")
    p.add_argument("--n-points", type=int, default=50)
    p.add_argument("--Ly", type=float, default=0.1,
                   help="Periodic domain width [m]")
    p.add_argument("--shock-pert-amp", type=float, default=1e-3,
                   help="Shock position perturbation amplitude [m]")
    p.add_argument("--v-pert-amp", type=float, default=40.0,
                   help="Transverse velocity perturbation amplitude [m/s]")
    p.add_argument("--x-shock", type=float, default=None,
                   help="Override shock position [m] (default: 1.5 * L_CJ)")
    p.add_argument("--cj-tol", type=float, default=0.01,
                   help="Tolerance for CJ distance detection (default 0.01)")
    p.add_argument("--pocket", action="store_true",
                   help="Add a rectangular pocket of unreacted gas behind the shock")
    p.add_argument("--pocket-offset", type=float, default=0.003,
                   help="Distance from shock to nearest pocket edge [m] (default 0.003)")
    p.add_argument("--pocket-size-x", type=float, default=0.01,
                   help="Pocket width in x [m] (default 0.01)")
    p.add_argument("--pocket-size-y", type=float, default=0.014,
                   help="Pocket height in y [m] (default 0.014)")
    p.add_argument("--pocket-T", type=float, default=2100.0,
                   help="Pocket temperature [K] (default 2100)")
    p.add_argument("--pocket-P", type=float, default=46667.0,
                   help="Pocket pressure [Pa] (default 46667)")
    p.add_argument("--pocket-y-center", type=float, default=None,
                   help="Pocket y-center [m] (default: Ly/2)")
    p.add_argument("--output", required=True, help="Output JSON file")
    args = p.parse_args()

    pocket_cfg = None
    if args.pocket:
        pocket_cfg = {
            "offset": args.pocket_offset,
            "size_x": args.pocket_size_x,
            "size_y": args.pocket_size_y,
            "T": args.pocket_T,
            "P": args.pocket_P,
        }
        if args.pocket_y_center is not None:
            pocket_cfg["y_center"] = args.pocket_y_center

    result = generate_exprtk(
        args.znd_profile,
        n_points=args.n_points,
        Ly=args.Ly,
        shock_pert_amp=args.shock_pert_amp,
        v_pert_amp=args.v_pert_amp,
        x_shock_override=args.x_shock,
        cj_tol=args.cj_tol,
        pocket=pocket_cfg,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)

    info = result["_info"]
    print(f"Generated exprtk initializer:")
    print(f"  L_CJ = {info['L_cj_mm']:.1f} mm")
    print(f"  x_shock = {info['x_shock_mm']:.1f} mm")
    print(f"  Interpolation points: {info['n_interp_points']}")
    print(f"  Species: {info['species_order']}")
    print(f"  Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
