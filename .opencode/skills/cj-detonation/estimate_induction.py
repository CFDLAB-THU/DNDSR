#!/usr/bin/env python3
"""Compute ignition delay and induction length from the von Neumann state.

Usage:
    python estimate_induction.py --mechanism h2o2.yaml --composition "H2:0.2, O2:0.1, AR:0.7" --basis mole
    python estimate_induction.py --mechanism h2o2.yaml --composition "H2:0.1111, O2:0.8889" --basis mass --U 2842

Requires the CJ speed to compute the post-shock state.  Use --U to specify
it directly, or the script runs estimate_cj internally to find it.
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
    p = argparse.ArgumentParser(
        description="Ignition delay / induction length estimator")
    p.add_argument("--mechanism", default="h2o2.yaml")
    p.add_argument("--composition", required=True,
                   help="e.g. 'H2:0.2, O2:0.1, AR:0.7'")
    p.add_argument("--basis", default="mole", choices=["mole", "mass"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=101325.0)
    p.add_argument("--U", type=float,
                   help="CJ detonation speed [m/s] (skips CJ search)")
    p.add_argument("--dx", type=float,
                   help="Mesh cell size [m] for cell-count estimate")
    p.add_argument("--t-max", type=float, default=1e-2,
                   help="Max integration time [s] (default 10ms)")
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

    # Get CJ speed
    U_CJ = args.U
    if U_CJ is None:
        # Quick CJ search
        U_range = np.linspace(500, 5000, 200)
        best = None
        for U in U_range:
            for vr in np.linspace(0.08, 0.40, 81):
                rho = rho1 / vr
                P = P1 + rho1 * U**2 * (1 - vr)
                h_target = h1 + 0.5 * U**2 * (1 - vr**2)
                if P <= 0:
                    continue
                try:
                    g2 = ct.Solution(args.mechanism)
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
                        best = (M2, U)
                except:
                    continue
        if best:
            U_CJ = best[1]
            print(f"CJ speed: {U_CJ:.1f} m/s (found internally)")
        else:
            print("ERROR: CJ speed not found. Supply --U.")
            return

    # von Neumann state
    T_vn, P_vn, rho_vn, u_vn = None, None, None, None
    MW1 = gas.mean_molecular_weight
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

    if T_vn is None:
        print("ERROR: von Neumann state not found.")
        return

    print(
        f"Post-shock (VN): T={T_vn:.0f} K, P={P_vn/1e5:.2f} bar, rho={rho_vn:.4f} kg/m3 ({rho_vn/rho1:.1f}x)")
    print(f"  u_particle = {u_vn:.0f} m/s")
    print(f"  U_CJ = {U_CJ:.1f} m/s")

    # Ignition delay via constant-pressure reactor
    gas3 = ct.Solution(args.mechanism)
    gas3.TPX = T_vn, P_vn, {k: v for k, v in comp.items()}

    reactor = ct.IdealGasConstPressureReactor(gas3, clone=False)
    net = ct.ReactorNet([reactor])

    t_ign = None
    T_ign = None
    OH_idx = gas3.species_index("OH") if "OH" in gas3.species_names else None

    for t in np.logspace(-10, np.log10(args.t_max), 20000):
        try:
            net.advance(t)
        except Exception:
            break
        Tcur = reactor.phase.T
        if Tcur > T_vn * 1.5 and t_ign is None:
            t_ign = t
            T_ign = Tcur
            break

    if t_ign is not None:
        d_induction = u_vn * t_ign
        print(f"\nIgnition: delay = {t_ign*1e6:.2f} μs")
        print(
            f"  Induction length = {d_induction*1e6:.0f} μm = {d_induction*1e3:.2f} mm")
        if args.dx:
            cells = d_induction / args.dx
            print(
                f"  Cells across induction zone = {cells:.0f} (dx = {args.dx*1e6:.0f} μm)")
        print(f"  T at ignition = {T_ign:.0f} K")
    else:
        print(
            f"\nIgnition not detected within {args.t_max*1e3:.0f} ms at T_VN={T_vn:.0f} K")


if __name__ == "__main__":
    main()
