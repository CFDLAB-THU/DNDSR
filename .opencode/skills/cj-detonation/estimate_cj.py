#!/usr/bin/env python3
"""Estimate CJ detonation speed and post-shock/CJ equilibrium states.

Uses SDToolbox (https://shepherd.caltech.edu/EDL/PublicResources/sdt/) for
reliable CJ speed computation via the minimum wave speed method.

Usage:
    python estimate_cj.py --mechanism gri30.yaml --composition "H2:2, O2:1"
    python estimate_cj.py --mechanism gri30.yaml \\
        --composition "H2:2, O2:1, AR:7" --basis mole \\
        --temperature 300 --pressure 101325
    python estimate_cj.py --mechanism h2o2.yaml \\
        --composition "H2:0.1111, O2:0.8889" --basis mass

Install SDToolbox:
    pip install "sdtoolbox @ git+https://github.com/harryzhou2000/sdtoolbox.git#subdirectory=Python3"
"""

from __future__ import annotations

import argparse
import numpy as np

import cantera as ct
from sdtoolbox.postshock import CJspeed, PostShock_eq, PostShock_fr
from sdtoolbox.thermo import soundspeed_eq, soundspeed_fr


def parse_composition(comp: str) -> str:
    """Convert "H2:2, O2:1" to "H2:2 O2:1" for SDToolbox."""
    return comp.replace(",", " ").strip()


def main():
    p = argparse.ArgumentParser(
        description="CJ detonation estimator (SDToolbox)")
    p.add_argument("--mechanism", default="gri30.yaml")
    p.add_argument("--composition", required=True,
                   help="e.g. 'H2:2, O2:1' or 'H2:2, O2:1, AR:7'")
    p.add_argument("--basis", default="mole", choices=["mole", "mass"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=101325.0)
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
    MW1 = gas1.mean_molecular_weight
    gamma1 = gas1.cp / gas1.cv
    a1 = np.sqrt(gamma1 * ct.gas_constant / MW1 * T1)

    print(f"Reactants: rho={rho1:.4g} kg/m3, T={T1:.1f} K, "
          f"P={P1/1e5:.4f} bar, a={a1:.0f} m/s, MW={MW1:.1f} g/mol")
    print(f"  Composition: {args.composition}")
    print(f"  Mechanism: {mech}")

    # CJ speed (minimum wave speed method)
    cj_speed = CJspeed(P1, T1, q, mech)

    # CJ equilibrium state
    gas_cj = PostShock_eq(cj_speed, P1, T1, q, mech)
    rho_cj = gas_cj.density
    ae = soundspeed_eq(gas_cj)
    w2 = cj_speed * rho1 / rho_cj  # particle velocity in wave frame
    u2 = cj_speed - w2              # particle velocity in lab frame
    M_product = abs(cj_speed - u2) / ae

    print(f"\nCJ detonation: U={cj_speed:.1f} m/s, M_product={M_product:.4f}")
    print(f"  T_CJ={gas_cj.T:.0f} K, P_CJ={gas_cj.P/1e5:.2f} bar, "
          f"rho_CJ={rho_cj:.4f} kg/m3 ({rho_cj/rho1:.2f}x)")

    # von Neumann (frozen shock) state
    gas_vn = PostShock_fr(cj_speed, P1, T1, q, mech)
    rho_vn = gas_vn.density
    u_vn = cj_speed * (1 - rho1 / rho_vn)
    a_vn = soundspeed_fr(gas_vn)
    M_vn = abs(cj_speed - u_vn) / a_vn

    print(f"\nvon Neumann spike (frozen shock):")
    print(f"  T_VN={gas_vn.T:.0f} K, P_VN={gas_vn.P/1e5:.2f} bar, "
          f"rho_VN={rho_vn:.4f} kg/m3 ({rho_vn/rho1:.2f}x)")
    print(f"  u_VN (particle vel)={u_vn:.0f} m/s")
    print(f"  a_VN={a_vn:.0f} m/s, M_VN={M_vn:.2f}")

    # Mesh resolution hint
    print(f"\nTimestep guide (U0 = reference velocity scale):")
    print(f"  dt_code ≈ dx / U_CJ * U0")
    print(f"  For dx=10μm: dt_code ≈ 1e-5 / {cj_speed:.0f} * U0")


if __name__ == "__main__":
    main()
