#!/usr/bin/env python3
"""Generate h2o2.yaml with thermo extended to 1 K using constant-Cp NASA9 polynomials.

The original NASA7 segments are converted to NASA9 and preserved exactly.
A new constant-Cp segment is prepended from 1 K to the original lower bound.
"""

import copy
import math
import sys
from pathlib import Path

import cantera as ct
import yaml

ORIG_MECH = Path(__file__).resolve(
).parents[2] / "external/cfd_externals/repos/cantera/data/h2o2.yaml"
OUT_MECH = Path(__file__).parent / "h2o2.yaml"
PHASE = "ohmech"


def nasa7_to_nasa9(coeffs: list[float]) -> list[float]:
    """Convert NASA7 coefficients to NASA9 format (extra 1/T^2 and 1/T terms are zero)."""
    return [0.0, 0.0] + list(coeffs)


def const_cp_nasa9(T_break: float, cp_Tb: float, h_Tb: float, s_Tb: float) -> list[float]:
    """Build a constant-Cp NASA9 segment matching (h,s,cp) at T_break.

    Parameters
    ----------
    T_break : float
        Junction temperature [K].
    cp_Tb : float
        Cp/R at T_break.
    h_Tb : float
        H/(R·T_break) at T_break.
    s_Tb : float
        S/R at T_break.

    Returns
    -------
    list[float]
        Nine coefficients [a1..a9] for the constant-Cp segment.
    """
    R = ct.gas_constant  # J/kmol·K, only used when passing raw molar values
    a3 = cp_Tb  # Cp/R constant
    a8 = T_break * (h_Tb - a3)
    a9 = s_Tb - a3 * math.log(T_break)
    return [0.0, 0.0, a3, 0.0, 0.0, 0.0, 0.0, a8, a9]


def main():
    # ------------------------------------------------------------------
    # 1. Load original mechanism (YAML for editing, Cantera for property evaluation)
    # ------------------------------------------------------------------
    with open(ORIG_MECH) as fh:
        doc = yaml.safe_load(fh)

    gas_orig = ct.Solution(str(ORIG_MECH), PHASE)

    # ------------------------------------------------------------------
    # 2. Build a per-species map of Cantera index for fast lookups
    # ------------------------------------------------------------------
    name_to_idx = {gas_orig.species_name(
        i): i for i in range(gas_orig.n_species)}

    # ------------------------------------------------------------------
    # 3. Convert each species
    # ------------------------------------------------------------------
    n_converted = 0
    for sp_entry in doc["species"]:
        name = sp_entry["name"]
        thermo = sp_entry["thermo"]

        if thermo["model"] != "NASA7":
            continue

        T_ranges = list(thermo["temperature-ranges"])  # [T_low, T_mid, T_high]
        T_orig_low = T_ranges[0]  # e.g. 200 or 300
        T_mid = T_ranges[1]        # e.g. 1000
        T_high = T_ranges[2]       # e.g. 3500 or 5000

        # Evaluate original properties at the junction temperature (T_orig_low)
        idx = name_to_idx[name]
        gas_orig.TPX = T_orig_low, ct.one_atm, {name: 1.0}

        cp_Tb = gas_orig.partial_molar_cp[idx] / ct.gas_constant
        h_Tb = gas_orig.partial_molar_enthalpies[idx] / \
            (ct.gas_constant * T_orig_low)
        s_Tb = gas_orig.partial_molar_entropies[idx] / ct.gas_constant

        # Convert existing segments to NASA9
        nasa9_low = nasa7_to_nasa9(thermo["data"][0])   # current low range
        nasa9_high = nasa7_to_nasa9(thermo["data"][1])  # current high range

        # Build the new ultra-low constant-Cp segment
        nasa9_ultra = const_cp_nasa9(T_orig_low, cp_Tb, h_Tb, s_Tb)

        # Update the YAML entry
        thermo["model"] = "NASA9"
        thermo["temperature-ranges"] = [1.0, T_orig_low, T_mid, T_high]
        thermo["data"] = [nasa9_ultra, nasa9_low, nasa9_high]

        n_converted += 1

    print(
        f"Converted {n_converted} species from NASA7 to NASA9 with 1 K extension.")

    # ------------------------------------------------------------------
    # 4. Convert all numeric values to plain Python floats (Cantera returns numpy scalars)
    # ------------------------------------------------------------------
    def deep_convert(obj):
        if isinstance(obj, float) and hasattr(obj, "item"):
            return float(obj)
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(i) for i in obj]
        return obj

    doc = deep_convert(doc)

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    with open(OUT_MECH, "w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=None,
                       sort_keys=False, width=120)

    print(f"Written {OUT_MECH}")

    # ------------------------------------------------------------------
    # 5. Validation
    # ------------------------------------------------------------------
    print("\n=== Validation ===")

    gas_new = ct.Solution(str(OUT_MECH), PHASE)

    # Pre-build a map: name -> (T_orig_low, T_high, T_mid)
    sp_bounds = {}
    for sp_entry in doc["species"]:
        tr = sp_entry["thermo"]["temperature-ranges"]
        sp_bounds[sp_entry["name"]] = (tr[1], tr[-1])  # (T_orig_low, T_high)

    # Check that original range (T_orig_low .. T_high) is unchanged
    test_temps = [200.0, 298.15, 500.0, 800.0, 1000.0,
                  1500.0, 2000.0, 3000.0, 3500.0, 5000.0]

    max_err_cp = 0.0
    max_err_h = 0.0
    max_err_s = 0.0

    for name in gas_orig.species_names:
        idx = name_to_idx[name]
        T_orig_low, T_high = sp_bounds[name]

        for T in test_temps:
            # Skip temps outside the original valid range -- those changed by design
            if T < T_orig_low or T > T_high:
                continue

            gas_orig.TPX = T, ct.one_atm, {name: 1.0}
            gas_new.TPX = T, ct.one_atm, {name: 1.0}

            cp_o = gas_orig.partial_molar_cp[idx]
            cp_n = gas_new.partial_molar_cp[idx]
            h_o = gas_orig.partial_molar_enthalpies[idx]
            h_n = gas_new.partial_molar_enthalpies[idx]
            s_o = gas_orig.partial_molar_entropies[idx]
            s_n = gas_new.partial_molar_entropies[idx]

            err_cp = abs(cp_n - cp_o) / max(1.0, abs(cp_o))
            err_h = abs(h_n - h_o) / max(1.0, abs(h_o))
            err_s = abs(s_n - s_o) / max(1.0, abs(s_o))

            max_err_cp = max(max_err_cp, err_cp)
            max_err_h = max(max_err_h, err_h)
            max_err_s = max(max_err_s, err_s)

            if err_cp > 1e-12 or err_h > 1e-12 or err_s > 1e-12:
                print(
                    f"  MISMATCH {name} @ T={T:.1f}: cp_err={err_cp:.2e} h_err={err_h:.2e} s_err={err_s:.2e}")

    print(
        f"  Max relative error in original range: cp={max_err_cp:.2e} h={max_err_h:.2e} s={max_err_s:.2e}")

    # Check low-T extension: constant Cp and finite values
    print("\n  Low-T extension check:")
    for name in gas_orig.species_names:
        idx = name_to_idx[name]
        T_orig_low = sp_bounds[name][0]

        gas_new.TPX = 1.0, ct.one_atm, {name: 1.0}
        cp_1K = gas_new.partial_molar_cp[idx]
        gas_new.TPX = 10.0, ct.one_atm, {name: 1.0}
        cp_10K = gas_new.partial_molar_cp[idx]
        gas_new.TPX = T_orig_low, ct.one_atm, {name: 1.0}
        cp_bp = gas_new.partial_molar_cp[idx]

        # Constant Cp means cp_1K == cp_10K == cp at break
        const_ok = abs(cp_1K - cp_10K) / max(1.0, abs(cp_10K)) < 1e-12
        bridge_ok = abs(cp_1K - cp_bp) / max(1.0, abs(cp_bp)) < 1e-12
        status = "OK" if (const_ok and bridge_ok and cp_1K > 0) else "FAIL"
        print(
            f"    {name:6s}: Cp@1K={cp_1K:.3f} Cp@10K={cp_10K:.3f} Cp@Tlow({T_orig_low:.0f})={cp_bp:.3f}  [{status}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
