#!/usr/bin/env python3
"""Estimate CJ detonation speed and post-shock/CJ equilibrium states.

Usage:
    python estimate_cj.py --mechanism h2o2.yaml --composition "H2:0.2, O2:0.1, AR:0.7" --basis mole
    python estimate_cj.py --mechanism h2o2.yaml --composition "H2:0.1111, O2:0.8889" --basis mass

If --U is given, skips the CJ search and just prints the post-shock state
at that speed.
"""

from __future__ import annotations

import argparse
import cantera as ct
import numpy as np


def parse_composition(comp: str, basis: str) -> dict[str, float]:
    result = {}
    for part in comp.split(","):
        k, v = part.strip().split(":")
        result[k.strip()] = float(v.strip())
    return result


def main():
    p = argparse.ArgumentParser(description="CJ detonation estimator")
    p.add_argument("--mechanism", default="h2o2.yaml")
    p.add_argument("--composition", required=True,
                   help="e.g. 'H2:0.2, O2:0.1, AR:0.7'")
    p.add_argument("--basis", default="mole", choices=["mole", "mass"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=101325.0)
    p.add_argument("--U", type=float, help="Skip CJ search, use this speed")
    p.add_argument("--U-min", type=float, default=500)
    p.add_argument("--U-max", type=float, default=5000)
    p.add_argument("--U-steps", type=int, default=200)
    args = p.parse_args()

    comp = parse_composition(args.composition, args.basis)
    gas = ct.Solution(args.mechanism)
    if args.basis == "mole":
        gas.TPX = args.temperature, args.pressure, comp
    else:
        gas.TPY = args.temperature, args.pressure, comp

    rho1 = gas.density
    P1 = gas.P
    T1 = gas.T
    h1 = gas.enthalpy_mass
    MW1 = gas.mean_molecular_weight
    gamma = gas.cp / gas.cv
    a1 = np.sqrt(gamma * ct.gas_constant / MW1 * T1)

    print(
        f"Reactants: rho={rho1:.4g} kg/m3, T={T1:.1f} K, P={P1/1e5:.4f} bar, a={a1:.0f} m/s, MW={MW1:.1f} g/mol")
    print(f"  Composition: {comp}")
    print(f"  Mechanism: {args.mechanism}")

    if args.U:
        U_list = [args.U]
    else:
        U_list = np.linspace(args.U_min, args.U_max, args.U_steps)

    # --- CJ search ---
    if not args.U:
        best = None
        for U in U_list:
            for vr in np.linspace(0.08, 0.40, 101):
                rho = rho1 / vr
                P = P1 + rho1 * U**2 * (1 - vr)
                h_target = h1 + 0.5 * U**2 * (1 - vr**2)
                if P <= 0:
                    continue
                try:
                    g2 = ct.Solution(args.mechanism)
                    # Set composition first, then use HP equilibrate
                    if args.basis == "mole":
                        g2.TPX = 2000.0, P, comp
                    else:
                        g2.TPY = 2000.0, P, comp
                    g2.HP = h_target, P
                    g2.equilibrate("HP")
                    if abs(g2.enthalpy_mass - h_target) / abs(h_target) > 0.02:
                        continue
                    u2 = rho1 * U / rho
                    a2 = np.sqrt(g2.cp / g2.cv * ct.gas_constant /
                                 g2.mean_molecular_weight * g2.T)
                    M2 = abs(U - u2) / a2
                    if best is None or abs(M2 - 1.0) < abs(best[0] - 1.0):
                        best = (M2, U, g2.T, g2.P, rho, u2)
                except:
                    continue
        if best is None:
            print("\nCJ point not found in U range.")
            return
        M2_best, U_CJ, T_CJ, P_CJ, rho_CJ, u2_CJ = best
        print(f"\nCJ detonation: U={U_CJ:.1f} m/s, M_product={M2_best:.4f}")
        print(
            f"  T_CJ={T_CJ:.0f} K, P_CJ={P_CJ/1e5:.2f} bar, rho_CJ={rho_CJ:.4f} kg/m3 ({rho_CJ/rho1:.2f}x)")
    else:
        U_CJ = args.U

    # --- von Neumann (frozen shock) state ---
    T_vn, P_vn, rho_vn, u_vn = None, None, None, None
    for vr in np.linspace(0.08, 0.50, 300):
        rho = rho1 / vr
        P = P1 + rho1 * U_CJ**2 * (1 - vr)
        Ttry = P / (rho * ct.gas_constant / MW1)
        if 500 < Ttry < 5000:
            try:
                g2 = ct.Solution(args.mechanism)
                g2.TPX = Ttry, P, {k: v for k, v in comp.items()}
                h_post = g2.enthalpy_mass
                h_targ = h1 + 0.5 * U_CJ**2 * (1 - vr**2)
                if abs(h_post - h_targ) / abs(h_targ) < 0.005:
                    T_vn, P_vn, rho_vn = Ttry, P, rho
                    u_vn = U_CJ * (1 - rho1 / rho_vn)
                    break
            except:
                continue

    if T_vn:
        print(f"\nvon Neumann spike (frozen shock):")
        print(
            f"  T_VN={T_vn:.0f} K, P_VN={P_vn/1e5:.2f} bar, rho_VN={rho_vn:.4f} kg/m3 ({rho_vn/rho1:.2f}x)")
        print(f"  u_VN (particle vel)={u_vn:.0f} m/s")
        a_vn = np.sqrt(gamma * ct.gas_constant / MW1 * T_vn)
        M_vn = abs(U_CJ - u_vn) / a_vn
        print(f"  a_VN={a_vn:.0f} m/s, M_VN={M_vn:.2f}")

    # --- Mesh resolution hint ---
    print(f"\nTimestep guide (U0 = reference velocity scale):")
    print(f"  dt_code ≈ dx / U_CJ * U0")
    print(f"  For dx=10μm: dt_code ≈ 1e-5 / {U_CJ:.0f} * U0")


if __name__ == "__main__":
    main()
