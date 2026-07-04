#!/usr/bin/env python3
"""Cantera 1D H2/O2 premixed flame references for DNDSR.

The runs here intentionally use mixture-averaged transport without Soret
diffusion. DNDSR's current reactive-flow transport does not include
multicomponent diffusion or thermal diffusion, so these cases are better
reference targets than the full upstream Cantera example.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cantera as ct
import numpy as np


@dataclass(frozen=True)
class FlameCase:
    name: str
    composition: str
    composition_basis: str
    temperature: float
    pressure: float
    width: float
    ratio: float
    slope: float
    curve: float


CASES = {
    "cantera_h2_o2_ar": FlameCase(
        name="cantera_h2_o2_ar",
        composition="H2:1.1, O2:1, AR:5",
        composition_basis="mole",
        temperature=300.0,
        pressure=ct.one_atm,
        width=0.03,
        ratio=3.0,
        slope=0.06,
        curve=0.12,
    ),
    "react_test_stoich_air": FlameCase(
        name="react_test_stoich_air",
        composition="H2:0.028, O2:0.222, N2:0.75",
        composition_basis="mass",
        temperature=300.0,
        pressure=ct.one_atm,
        width=0.03,
        ratio=3.0,
        slope=0.06,
        curve=0.12,
    ),
}


def set_gas_state(gas: ct.Solution, case: FlameCase) -> None:
    if case.composition_basis == "mole":
        gas.TPX = case.temperature, case.pressure, case.composition
    elif case.composition_basis == "mass":
        gas.TPY = case.temperature, case.pressure, case.composition
    else:
        raise ValueError(
            f"unknown composition basis: {case.composition_basis}")


def solve_case(case: FlameCase, mechanism: str, loglevel: int) -> tuple[ct.FreeFlame, dict]:
    gas = ct.Solution(mechanism)
    set_gas_state(gas, case)

    inlet_y = dict(zip(gas.species_names, gas.Y))
    inlet_x = dict(zip(gas.species_names, gas.X))
    inlet_density = gas.density

    flame = ct.FreeFlame(gas, width=case.width)
    flame.set_refine_criteria(
        ratio=case.ratio, slope=case.slope, curve=case.curve)
    flame.transport_model = "mixture-averaged"
    flame.soret_enabled = False
    flame.flux_gradient_basis = "mass"
    flame.solve(loglevel=loglevel, auto=True)

    summary = summarize(case, mechanism, flame,
                        inlet_density, inlet_x, inlet_y)
    return flame, summary


def summarize(
    case: FlameCase,
    mechanism: str,
    flame: ct.FreeFlame,
    inlet_density: float,
    inlet_x: dict[str, float],
    inlet_y: dict[str, float],
) -> dict:
    grid = np.asarray(flame.grid)
    temperature = np.asarray(flame.T)
    dtdx = np.gradient(temperature, grid)
    max_dtdx = float(np.max(dtdx))
    thermal_thickness = float((temperature[-1] - temperature[0]) / max_dtdx)

    return {
        "case": asdict(case),
        "mechanism": mechanism,
        "cantera_version": ct.__version__,
        "transport_model": flame.transport_model,
        "soret_enabled": bool(flame.soret_enabled),
        "flux_gradient_basis": flame.flux_gradient_basis,
        "flame_speed_m_per_s": float(flame.velocity[0]),
        "inlet_mass_flux_kg_per_m2_s": float(inlet_density * flame.velocity[0]),
        "inlet_density_kg_per_m3": float(inlet_density),
        "burned_temperature_K": float(temperature[-1]),
        "domain_width_m": float(grid[-1] - grid[0]),
        "grid_points": int(grid.size),
        "thermal_thickness_m": thermal_thickness,
        "max_temperature_gradient_K_per_m": max_dtdx,
        "inlet_mole_fractions": inlet_x,
        "inlet_mass_fractions": inlet_y,
        "outlet_mass_fractions": dict(zip(flame.gas.species_names, flame.Y[:, -1])),
    }


def write_profile_csv(path: Path, flame: ct.FreeFlame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    species = flame.gas.species_names
    fields = [
        "z_m",
        "T_K",
        "p_Pa",
        "rho_kg_per_m3",
        "velocity_m_per_s",
        "heat_release_W_per_m3",
    ]
    fields.extend(f"Y_{name}" for name in species)

    y = np.asarray(flame.Y).T
    with path.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(fields)
        for i, z in enumerate(flame.grid):
            row = [
                z,
                flame.T[i],
                flame.P,
                flame.density[i],
                flame.velocity[i],
                flame.heat_release_rate[i],
            ]
            row.extend(y[i, :])
            writer.writerow(row)


def save_restore_file(path: Path, flame: ct.FreeFlame, case_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    flame.save(
        path,
        name=case_name,
        description="DNDSR reference: mixture-averaged transport, no Soret",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", *CASES.keys()],
        default="all",
        help="case preset to run",
    )
    parser.add_argument(
        "--mechanism",
        default="h2o2.yaml",
        help="Cantera mechanism path or name",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reference",
        help="directory for CSV profiles, restore files, and summaries",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=1,
        help="Cantera 1D solver verbosity, 0 to 8",
    )
    return parser.parse_args()


def selected_cases(case_name: str) -> Iterable[FlameCase]:
    if case_name == "all":
        return CASES.values()
    return (CASES[case_name],)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case in selected_cases(args.case):
        flame, summary = solve_case(case, args.mechanism, args.loglevel)
        summaries.append(summary)

        write_profile_csv(args.out_dir / f"{case.name}_profile.csv", flame)
        save_restore_file(args.out_dir / f"{case.name}.yaml", flame, case.name)

        print(
            f"{case.name}: Su={summary['flame_speed_m_per_s']:.9g} m/s, "
            f"T_b={summary['burned_temperature_K']:.6g} K, "
            f"points={summary['grid_points']}"
        )

    summary_path = args.out_dir / "h2o2_free_flame_mixture_averaged_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
