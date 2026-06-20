#!/usr/bin/env python3
"""Compute ignition delay and induction length via ZND detonation structure.

Uses SDToolbox (https://shepherd.caltech.edu/EDL/PublicResources/sdt/) for the
ZND model — integrates the reactive Euler equations through the detonation wave
starting from the von Neumann (frozen post-shock) state.

The induction length is defined as the distance from the shock to the point
of maximum thermicity gradient (dσ̇/dx).  The exothermic length is the
half-width of the thermicity pulse.

Usage:
    python estimate_induction.py --mechanism gri30.yaml --composition "H2:2, O2:1"
    python estimate_induction.py --mechanism gri30.yaml \\
        --composition "H2:2, O2:1, AR:7" --basis mole \\
        --temperature 300 --pressure 6670 --dx 1e-5 --overdrive 1.001

Install SDToolbox:
    pip install "sdtoolbox @ git+https://github.com/harryzhou2000/sdtoolbox.git#subdirectory=Python3"
"""

from __future__ import annotations

import argparse
import sys

import cantera as ct
import numpy as np
from sdtoolbox.postshock import CJspeed, PostShock_fr
from sdtoolbox.znd import zndsolve


def plot_znd_profile(znd_out, gas, cj_speed, overdrive, plot_path,
                     ind_len=None, exo_len=None):
    import matplotlib.pyplot as plt

    dist = znd_out["distance"] * 1e3  # mm
    T = znd_out["T"]
    P = znd_out["P"] / 1e5  # bar
    rho = znd_out["rho"]
    therm = znd_out["thermicity"]
    species = znd_out["species"]
    M = znd_out["M"]

    species_names = gas.species_names
    n_species = len(species_names)

    if ind_len is not None:
        reaction_scale = ind_len + (exo_len if exo_len else ind_len)
        xlim_mm = max(10 * ind_len * 1e3, 3 * reaction_scale * 1e3)
    else:
        xlim_mm = dist[-1]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(dist, T, "r-", linewidth=1.5)
    axes[0].set_ylabel("Temperature [K]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dist, P, "b-", linewidth=1.5)
    axes[1].set_ylabel("Pressure [bar]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dist, therm, "k-", linewidth=1.5)
    axes[2].set_ylabel("Thermicity [1/s]")
    axes[2].set_yscale("symlog", linthresh=1e3)
    axes[2].grid(True, alpha=0.3)

    major_threshold = 0.01
    for i in range(n_species):
        y = species[i] if species.shape[0] == n_species else species[:, i]
        if np.max(y) > major_threshold:
            axes[3].plot(dist, y, linewidth=1.2, label=species_names[i])
    axes[3].set_ylabel("Mass fraction")
    axes[3].set_xlabel("Distance from shock [mm]")
    axes[3].legend(loc="center right", fontsize=8, ncol=2)
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, xlim_mm)
        if ind_len is not None:
            ax.axvline(ind_len * 1e3, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.7)

    title = (f"ZND Profile — U={overdrive * cj_speed:.0f} m/s "
             f"(U_CJ={cj_speed:.0f}, f={overdrive:.3f})")
    if ind_len is not None:
        title += f"  |  ℓ_ind={ind_len*1e3:.3f} mm"
    if exo_len is not None:
        title += f", ℓ_exo={exo_len*1e3:.3f} mm"
    axes[0].set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nZND profile plot saved to {plot_path}")


def parse_composition(comp: str) -> str:
    """Convert "H2:2, O2:1" to "H2:2 O2:1" for SDToolbox."""
    return comp.replace(",", " ").strip()


def main():
    p = argparse.ArgumentParser(
        description="Ignition delay / induction length estimator (SDToolbox ZND)")
    p.add_argument("--mechanism", default="gri30.yaml")
    p.add_argument("--composition", required=True,
                   help="e.g. 'H2:2, O2:1' or 'H2:2, O2:1, AR:7'")
    p.add_argument("--basis", default="mole", choices=["mole", "mass"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=101325.0)
    p.add_argument("--overdrive", type=float, default=1.001,
                   help="U/U_CJ > 1 to avoid sonic singularity (default 1.001)")
    p.add_argument("--t-end", type=float, default=None,
                   help="Max integration time [s] (auto if not set)")
    p.add_argument("--dx", type=float,
                   help="Mesh cell size [m] for cell-count estimate")
    p.add_argument("--tolerance", type=float, default=1e-8,
                   help="Integration tolerance (relTol=absTol=tolerance, default 1e-8)")
    p.add_argument("--znd-output", type=str,
                   help="Save full ZND profile to this file (npz format)")
    p.add_argument("--plot-output", type=str,
                   help="Save ZND profile plot to this file (png/pdf/svg)")
    args = p.parse_args()

    q = parse_composition(args.composition)
    P1 = args.pressure
    T1 = args.temperature
    mech = args.mechanism

    # Initial state
    gas1 = ct.Solution(mech)
    if args.basis == "mole":
        gas1.TPX = T1, P1, q
    else:
        gas1.TPY = T1, P1, q
    rho1 = gas1.density
    a1 = np.sqrt(gas1.cp / gas1.cv * ct.gas_constant /
                 gas1.mean_molecular_weight * T1)

    # CJ speed
    cj_speed = CJspeed(P1, T1, q, mech)
    U = args.overdrive * cj_speed

    # VN state at the overdriven speed
    gas = PostShock_fr(U, P1, T1, q, mech)

    print(f"CJ speed: {cj_speed:.1f} m/s")
    print(f"  Overdrive factor: {args.overdrive:.3f} -> U={U:.1f} m/s")

    # Auto t_end if not set
    if args.t_end is None:
        # Estimate from CJ speed and typical induction scale
        t_end_defaults = {101325: 1e-4, 6670: 1e-2, 10132: 5e-3}
        args.t_end = max(1e-5, min(t_end_defaults.get(int(P1), 1e-3), 1e-1))
        print(f"  Auto t_end = {args.t_end:.1e} s")

    # ZND solve
    print(f"Post-shock (VN): T={gas.T:.0f} K, P={gas.P/1e5:.2f} bar, "
          f"rho={gas.density:.4f} kg/m3 ({gas.density/rho1:.1f}x)")

    try:
        znd_out = zndsolve(
            gas, gas1, U,
            t_end=args.t_end,
            relTol=args.tolerance,
            absTol=args.tolerance,
            advanced_output=True,
            Method="LSODA",
        )
    except Exception as e:
        print(f"\nZND solve failed: {e}", file=sys.stderr)
        print("Try increasing --overdrive (e.g. 1.005) or --t-end.", file=sys.stderr)
        return 1

    u_vn = U * (1 - rho1 / gas.density)
    ind_len = znd_out["ind_len_ZND"]
    ind_time = znd_out["ind_time_ZND"]
    exo_len = znd_out["exo_len_ZND"]
    xfinal = znd_out["xfinal"]

    print(f"\nInduction zone (max thermicity gradient):")
    print(f"  τ_induction = {ind_time*1e6:.2f} μs")
    print(f"  ℓ_induction = {ind_len*1e3:.2f} mm ({ind_len*1e6:.0f} μm)")
    print(f"  ℓ_exothermic = {exo_len*1e3:.2f} mm ({exo_len*1e6:.0f} μm)")
    print(f"  Total ZND length (xfinal) = {xfinal*1e3:.2f} mm")
    print(f"  Ratio exo/ind = {exo_len/ind_len:.2f}")

    if args.dx:
        cells_ind = ind_len / args.dx
        cells_total = xfinal / args.dx
        print(f"\nMesh resolution (dx = {args.dx*1e6:.0f} μm):")
        print(f"  Cells across induction zone = {cells_ind:.0f}")
        print(f"  Cells across full ZND zone = {cells_total:.0f}")
        print(f"  dt_code ≈ dx / U_CJ * U0 = {args.dx/cj_speed:.3e} * U0")

    if args.znd_output:
        np.savez(args.znd_output,
                 distance=znd_out["distance"],
                 time=znd_out["time"],
                 T=znd_out["T"],
                 P=znd_out["P"],
                 rho=znd_out["rho"],
                 U_loc=znd_out["U"],
                 thermicity=znd_out["thermicity"],
                 species=znd_out["species"],
                 M=znd_out["M"])
        print(f"\nZND profile saved to {args.znd_output}")

    if args.plot_output:
        plot_znd_profile(znd_out, gas, cj_speed, args.overdrive,
                         args.plot_output, ind_len=ind_len, exo_len=exo_len)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
