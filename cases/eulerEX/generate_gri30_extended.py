#!/usr/bin/env python3
"""Generate a GRI30 mechanism with constant-cp/cv thermo extended to 1 K."""

from __future__ import annotations

import math
from pathlib import Path

import cantera as ct
import yaml
from ruamel.yaml import YAML


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_MECHANISM = SCRIPT_DIR.parents[1] / \
    "external/cfd_externals/repos/cantera/data/gri30.yaml"
OUTPUT_MECHANISM = SCRIPT_DIR / "gri30.yaml"
PHASE_NAME = "gri30"
LOWER_TEMPERATURE_K = 1.0
RELATIVE_TOLERANCE = 2.0e-12


def nasa7_to_nasa9(coefficients: list[float]) -> list[float]:
    return [0.0, 0.0, *coefficients]


def constant_heat_capacity_nasa9(
    break_temperature: float,
    cp_over_r: float,
    h_over_rt: float,
    s_over_r: float,
) -> list[float]:
    enthalpy_constant = break_temperature * (h_over_rt - cp_over_r)
    entropy_constant = s_over_r - cp_over_r * math.log(break_temperature)
    return [
        0.0,
        0.0,
        float(cp_over_r),
        0.0,
        0.0,
        0.0,
        0.0,
        float(enthalpy_constant),
        float(entropy_constant),
    ]


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(expected))


def main() -> None:
    yaml12 = YAML(typ="safe")
    yaml12.version = (1, 2)
    document = yaml12.load(SOURCE_MECHANISM.read_text())
    source_gas = ct.Solution(str(SOURCE_MECHANISM), PHASE_NAME)
    source_indices = {name: source_gas.species_index(
        name) for name in source_gas.species_names}
    original_ranges: dict[str, list[float]] = {}

    for species in document["species"]:
        thermo = species["thermo"]
        if thermo["model"] != "NASA7":
            raise RuntimeError(
                f"unsupported thermo model for {species['name']}: {thermo['model']}")
        ranges = [float(value) for value in thermo["temperature-ranges"]]
        original_lower = ranges[0]
        original_upper = ranges[-1]
        species_index = source_indices[species["name"]]
        source_gas.TPX = original_lower, ct.one_atm, {species["name"]: 1.0}
        cp_over_r = source_gas.partial_molar_cp[species_index] / \
            ct.gas_constant
        h_over_rt = source_gas.partial_molar_enthalpies[species_index] / (
            ct.gas_constant * original_lower
        )
        s_over_r = source_gas.partial_molar_entropies[species_index] / \
            ct.gas_constant
        thermo["model"] = "NASA9"
        thermo["temperature-ranges"] = [LOWER_TEMPERATURE_K, *ranges]
        thermo["data"] = [
            constant_heat_capacity_nasa9(
                original_lower, cp_over_r, h_over_rt, s_over_r),
            *(nasa7_to_nasa9(coefficients) for coefficients in thermo["data"]),
        ]
        original_ranges[species["name"]] = ranges

    description = str(document.get("description", "")).rstrip()
    document["description"] = (
        description
        + "\n\nThermodynamic polynomials extended to 1 K by DNDSR: each added low-temperature "
        "NASA9 interval holds cp and cv constant at the species' original lower endpoint "
        "and matches cp, cv, h, and s continuously there."
    )
    OUTPUT_MECHANISM.write_text(
        yaml.safe_dump(document, default_flow_style=None,
                       sort_keys=False, width=120)
    )

    extended_gas = ct.Solution(str(OUTPUT_MECHANISM), PHASE_NAME)
    maximum_errors = {"cp": 0.0, "cv": 0.0, "h": 0.0, "s": 0.0}
    maximum_join_selection_errors = {"cp": 0.0, "cv": 0.0, "h": 0.0, "s": 0.0}
    low_temperature_failures: list[str] = []

    for species_name, ranges in original_ranges.items():
        original_lower = ranges[0]
        original_upper = ranges[-1]
        species_index = source_indices[species_name]
        sample_temperatures = {original_lower, original_upper}
        for lower, upper in zip(ranges[:-1], ranges[1:]):
            sample_temperatures.add(0.5 * (lower + upper))
        for join_temperature in ranges[1:-1]:
            offset = max(1.0e-8, join_temperature * 1.0e-10)
            sample_temperatures.update(
                (join_temperature - offset, join_temperature + offset))
        sample_temperatures = sorted(sample_temperatures)
        for temperature in sample_temperatures:
            source_gas.TPX = temperature, ct.one_atm, {species_name: 1.0}
            extended_gas.TPX = temperature, ct.one_atm, {species_name: 1.0}
            pairs = {
                "cp": (extended_gas.partial_molar_cp[species_index], source_gas.partial_molar_cp[species_index]),
                "cv": (extended_gas.partial_molar_cp[species_index] - ct.gas_constant, source_gas.partial_molar_cp[species_index] - ct.gas_constant),
                "h": (extended_gas.partial_molar_enthalpies[species_index], source_gas.partial_molar_enthalpies[species_index]),
                "s": (extended_gas.partial_molar_entropies[species_index], source_gas.partial_molar_entropies[species_index]),
            }
            for quantity, (actual, expected) in pairs.items():
                maximum_errors[quantity] = max(
                    maximum_errors[quantity], relative_error(actual, expected))

        for join_temperature in ranges[1:-1]:
            source_gas.TPX = join_temperature, ct.one_atm, {species_name: 1.0}
            extended_gas.TPX = join_temperature, ct.one_atm, {
                species_name: 1.0}
            join_pairs = {
                "cp": (extended_gas.partial_molar_cp[species_index], source_gas.partial_molar_cp[species_index]),
                "cv": (extended_gas.partial_molar_cp[species_index] - ct.gas_constant, source_gas.partial_molar_cp[species_index] - ct.gas_constant),
                "h": (extended_gas.partial_molar_enthalpies[species_index], source_gas.partial_molar_enthalpies[species_index]),
                "s": (extended_gas.partial_molar_entropies[species_index], source_gas.partial_molar_entropies[species_index]),
            }
            for quantity, (actual, expected) in join_pairs.items():
                maximum_join_selection_errors[quantity] = max(
                    maximum_join_selection_errors[quantity], relative_error(
                        actual, expected)
                )

        endpoint_values: dict[str, tuple[float, float, float]] = {}
        for temperature in (LOWER_TEMPERATURE_K, 10.0, original_lower):
            extended_gas.TPX = temperature, ct.one_atm, {species_name: 1.0}
            cp_value = extended_gas.partial_molar_cp[species_index]
            cv_value = cp_value - ct.gas_constant
            endpoint_values[str(temperature)] = (
                cp_value, cv_value, extended_gas.partial_molar_enthalpies[species_index])
        cp_1, cv_1, _ = endpoint_values[str(LOWER_TEMPERATURE_K)]
        cp_10, cv_10, _ = endpoint_values["10.0"]
        cp_endpoint, cv_endpoint, _ = endpoint_values[str(original_lower)]
        if not all(math.isfinite(value) for values in endpoint_values.values() for value in values):
            low_temperature_failures.append(
                f"{species_name}: non-finite low-temperature property")
        if cp_1 <= 0.0 or cv_1 <= 0.0:
            low_temperature_failures.append(
                f"{species_name}: non-positive heat capacity")
        if max(
            relative_error(cp_1, cp_10),
            relative_error(cp_1, cp_endpoint),
            relative_error(cv_1, cv_10),
            relative_error(cv_1, cv_endpoint),
        ) > RELATIVE_TOLERANCE:
            low_temperature_failures.append(
                f"{species_name}: cp/cv extension is not constant")

    if any(error > RELATIVE_TOLERANCE for error in maximum_errors.values()):
        raise RuntimeError(f"original-range thermo mismatch: {maximum_errors}")
    if low_temperature_failures:
        raise RuntimeError("; ".join(low_temperature_failures))

    print(f"source={SOURCE_MECHANISM}")
    print(f"output={OUTPUT_MECHANISM}")
    print(
        f"species={extended_gas.n_species} reactions={extended_gas.n_reactions}")
    print(f"minimum_temperature={extended_gas.min_temp:g} K")
    print(f"maximum_relative_errors={maximum_errors}")
    print(
        f"maximum_exact_join_selection_differences={maximum_join_selection_errors}")
    print("constant_cp_cv_extension=passed")


if __name__ == "__main__":
    main()
