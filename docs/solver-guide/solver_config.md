# Solver Configuration {#solver_config}

**Status:** Skeleton — content to be filled as the JSON schema stabilizes.

DNDSR solvers are controlled through JSON configuration files. The C++ solver
reads the JSON at startup; the Python `EulerP` module can also construct a
config programmatically.

> **TODO:** Add a complete JSON Schema or annotated example for each solver
> variant (`euler`, `eulerSA`, `euler2EQ`, `euler3D`, ...).

---

## File Layout

A typical solver config lives under `cases/<solver>/<case_name>.json`:

```json
{
  "meshFile": "path/to/mesh.cgns",
  "meshElevation": "",
  "meshDirectBisect": 0,
  "outputDirectory": "./output",
  "nTimeSteps": 1000,
  "dt": 1e-3,
  ...
}
```

> **TODO:** Document the top-level keys common to all solvers.

---

## Sections

### Mesh Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `meshFile` | `string` | required | Path to CGNS or H5 mesh file |
| `meshElevation` | `string` | `""` | `""` or `"O2"` for quadratic elevation |
| `meshDirectBisect` | `int` | `0` | Number of bisection levels |
| `periodicTranslation` | `array[3]` | — | Periodic translation vector |
| `periodicTolerance` | `float` | `1e-9` | Tolerance for periodic deduplication |

> **TODO:** Complete all mesh-related keys.

### Physics Settings

> **TODO:** Document keys for:
> - Equation set (`NS`, `NS_SA`, `NS_2EQ`, ...)
> - Gas model (`IdealGas`, ...)
> - Reynolds number, Mach number, reference conditions
> - Turbulence model parameters (SA, k-omega SST, etc.)
> - Rotation / source terms

### Numerical Settings

> **TODO:** Document keys for:
> - Spatial order (`P1`, `P2`, `P3`)
> - Time integrator (`SSPRK3`, `BDF2`, `ESDIRK4`, `HM3`, ...)
> - CFL number and adaptivity
> - Limiter settings
> - Jacobian update frequency
> - Linear solver settings (GMRES restart, tolerance, preconditioner)

### Boundary Conditions

Boundary conditions are specified as a list of objects, each matching a zone
name from the CGNS file:

```json
{
  "boundaryConditions": [
    {
      "name": "wall",
      "type": "Wall",
      "temperature": "adiabatic"
    },
    {
      "name": "farfield",
      "type": "Riemann",
      "state": "freestream"
    }
  ]
}
```

> **TODO:** List all supported BC types and their parameters.
> - Wall (no-slip, adiabatic, isothermal)
> - Inlet (total pressure/temperature, mass flow)
> - Outlet (static pressure, extrapolation)
> - Symmetry
> - Periodic
> - Riemann (far-field)

### Output Settings

> **TODO:** Document keys for:
> - VTK / Tecplot output frequency and fields
> - Restart checkpointing
> - Console logging level and frequency
> - Probe / slice / force extraction

---

## Solver-Specific Notes

### Euler (Inviscid)

> **TODO:** Keys unique to inviscid Euler solver.

### EulerSA (Spalart-Allmaras)

> **TODO:** SA-specific parameters (CB1, CB2, sigma, ...).

### Euler2EQ (k-omega / SST)

> **TODO:** Two-equation model-specific parameters.

---

## Complete Example

> **TODO:** Provide a fully commented example config for a standard test case
> (e.g. NACA0012, periodic vortex, or cylinder).

---

## See Also

- @ref building — how to build the solver executables
- @ref python_geom_guide — how to read and prepare meshes in Python
- @ref euler_unit_tests — regression test cases that exercise config parsing
