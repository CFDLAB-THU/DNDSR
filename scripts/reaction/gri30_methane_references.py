#!/usr/bin/env python3
"""Calculate methane flame and equilibrium states for DNDSR GRI30 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cantera as ct
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", type=Path,
                        default=Path("cases/eulerEX/gri30.yaml"))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--loglevel", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composition_map(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values) if value > 0.0}


def write_flame_profile(path: Path, flame: ct.FreeFlame) -> None:
    fields = ["x_m", "temperature_K", "pressure_Pa",
              "density_kg_m3", "velocity_m_s", "heat_release_W_m3"]
    fields.extend(f"Y_{name}" for name in flame.gas.species_names)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for index, coordinate in enumerate(flame.grid):
            writer.writerow(
                [
                    coordinate,
                    flame.T[index],
                    flame.P,
                    flame.density[index],
                    flame.velocity[index],
                    flame.heat_release_rate[index],
                    *flame.Y[:, index],
                ]
            )


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    mechanism = args.mechanism.resolve()

    gas = ct.Solution(str(mechanism), "gri30")
    gas.TPX = 300.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
    inlet_density = gas.density
    inlet_mass_fractions = composition_map(gas.species_names, gas.Y)
    flame = ct.FreeFlame(gas, width=0.04)
    flame.transport_model = "mixture-averaged"
    flame.soret_enabled = False
    flame.flux_gradient_basis = "mass"
    flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.12)
    flame.solve(loglevel=args.loglevel, auto=True)
    temperature = np.asarray(flame.T)
    gradient = np.gradient(temperature, np.asarray(flame.grid))
    flame_summary = {
        "model": "Cantera FreeFlame",
        "cantera_version": ct.__version__,
        "mechanism": str(mechanism),
        "mechanism_sha256": sha256(mechanism),
        "composition": "CH4:1, O2:2, N2:7.52",
        "composition_basis": "mole",
        "temperature_K": 300.0,
        "pressure_Pa": ct.one_atm,
        "transport_model": flame.transport_model,
        "soret_enabled": bool(flame.soret_enabled),
        "flux_gradient_basis": flame.flux_gradient_basis,
        "flame_speed_m_s": float(flame.velocity[0]),
        "burned_temperature_K": float(flame.T[-1]),
        "thermal_thickness_m": float((temperature[-1] - temperature[0]) / np.max(gradient)),
        "grid_points": int(flame.grid.size),
        "inlet_density_kg_m3": float(inlet_density),
        "inlet_mass_fractions": inlet_mass_fractions,
        "burned_mass_fractions": composition_map(flame.gas.species_names, flame.Y[:, -1]),
    }
    write_flame_profile(args.output_directory /
                        "methane_air_flame_profile.csv", flame)
    (args.output_directory /
     "methane_air_flame.json").write_text(json.dumps(flame_summary, indent=2) + "\n")

    equilibrium_gas = ct.Solution(str(mechanism), "gri30")
    equilibrium_gas.TPX = 3500.0, 20.0 * ct.one_atm, "CH4:1, O2:2"
    equilibrium_gas.equilibrate("TP")
    spark_summary = {
        "model": "Cantera constant-TP equilibrium spark state",
        "cantera_version": ct.__version__,
        "mechanism": str(mechanism),
        "mechanism_sha256": sha256(mechanism),
        "reactant_composition": "CH4:1, O2:2",
        "composition_basis": "mole",
        "temperature_K": float(equilibrium_gas.T),
        "pressure_Pa": float(equilibrium_gas.P),
        "mass_fractions": composition_map(equilibrium_gas.species_names, equilibrium_gas.Y),
    }
    (args.output_directory /
     "methane_oxygen_spark.json").write_text(json.dumps(spark_summary, indent=2) + "\n")
    print(json.dumps({"flame": flame_summary,
          "spark": spark_summary}, indent=2))


if __name__ == "__main__":
    main()
