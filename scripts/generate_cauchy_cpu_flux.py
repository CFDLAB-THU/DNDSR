#!/usr/bin/env python3
"""Emit compact AoS CPU flux instantiations and their physical shape registry."""
import argparse
from pathlib import Path

from generate_cauchy_rhs import cmake_index, read_shapes, write_if_changed


def generate(manifest, directory):
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "FiniteVolumeCPUFluxCatalog.hpp"
    registry = directory / "FiniteVolumeCPUFluxCatalog.cpp"
    index = directory / "FiniteVolumeRHSKernels_Sources.cmake"
    declarations, entries, sources = [], [], [registry]
    for acm in (False, True):
        shapes = read_shapes(manifest, acm)
        for c, m, d in shapes:
            profile = ("ACM" if acm else "PerfectGas") + ("", "SA", "TwoEquation")[c - (4 if acm else 5)]
            for form in (("Modal",) if m else ("MeanOnly", "DirectGradient")):
                name = f"Flux_{profile}_{form}_C{c}_M{m}_D{d}"
                tag = f"EquationPhysicsProfile::{profile}, ReconstructionForm::{form}"
                path = directory / f"FiniteVolumeCPUFluxCatalog_{name}.cpp"
                write_if_changed(path, f'''// Generated CPU specialization.
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels_Equation_Impl.cpp.inc"
namespace DNDS::Cauchy::CPU
{{
    NumericalResult<index> {name}(const FluxRowsRequest &a) noexcept
    {{
        return EvaluateEquationFluxRows<{tag}, {c}, {d}, {m}>(
            a.request, a.equation, a.boundary, a.flux, a.worker,
            a.owned.CompactRows<{c}>(), a.remote.CompactRows<{c}>(),
            a.destination.CompactRows<{c}>(), a.low.CompactRows<{c}>(), a.region,
            a.scratch, a.first, a.last, a.faces);
    }}
}}
''')
                sources.append(path)
                declarations.append(f"    NumericalResult<index> {name}(const FluxRowsRequest &) noexcept;")
                entries.append(f"            {{{tag}, {c}, {d}, {m}, {name}}},")
    write_if_changed(header, '''#pragma once
#include "Cauchy/Discretization/CPU/FiniteVolumeFluxShapes.hpp"
namespace DNDS::Cauchy::CPU
{
    FluxRowsKernel FindCPUFluxShape(EquationPhysicsProfile physics, ReconstructionForm form, index width, int dim, int modes);
}
''')
    write_if_changed(registry, '#include "FiniteVolumeCPUFluxCatalog.hpp"\nnamespace DNDS::Cauchy::CPU\n{\n'
                     + '\n'.join(declarations) + '''
    FluxRowsKernel FindCPUFluxShape(EquationPhysicsProfile physics, ReconstructionForm form, index width, int dim, int modes)
    {
        struct Entry { EquationPhysicsProfile physics; ReconstructionForm form; int width, dim, modes; FluxRowsKernel run; };
        static constexpr Entry entries[]{
''' + '\n'.join(entries) + '''
        };
        for (const auto &entry : entries)
            if (entry.physics == physics && entry.form == form && entry.width == width &&
                entry.dim == dim && entry.modes == modes)
                return entry.run;
        return nullptr;
    }
}
''')
    write_if_changed(index, cmake_index(directory, sources, sources + [header, index], header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate(args.manifest, args.output_dir.resolve())
