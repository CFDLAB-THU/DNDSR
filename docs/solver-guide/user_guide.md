# User Guide {#user_guide}

**Status:** Skeleton — to be expanded as tutorials are written.

This guide covers running DNDSR simulations from mesh generation through
to post-processing.

---

## Overview

DNDSR is a compressible Navier-Stokes solver using high-order Compact Finite
Volume (CFV) / Variational Reconstruction (VR) on unstructured meshes. It
supports:

- 2D and 3D inviscid and viscous flows
- RANS turbulence models (Spalart-Allmaras, k-omega SST, Realizable k-epsilon)
- MPI parallelism for multi-node clusters
- Optional CUDA GPU acceleration (EulerP module)

### Workflow

```
Mesh generation (Pointwise / Gmsh / other) → CGNS
       ↓
Solver configuration (JSON)
       ↓
Run solver (C++ executable or Python script)
       ↓
Post-processing (ParaView / VisIt / Tecplot)
```

---

## Mesh Generation

DNDSR reads CGNS format meshes. We currently use **Pointwise** for mesh
generation.

> **TODO:** Add recommended Pointwise export settings (CGNS version, zone
> organization, boundary condition naming).

### Supported Element Types

| Dimension | Linear (O1) | Quadratic (O2) |
|-----------|-------------|----------------|
| 2D | Tri3, Quad4 | Tri6, Quad9 |
| 3D | Tet4, Hex8, Prism6, Pyramid5 | Tet10, Hex27, Prism18, Pyramid14 |

> **TODO:** Notes on mesh quality requirements (skewness, aspect ratio,
> minimum cell size) for high-order VR.

---

## Running a Simulation

### C++ Solver

```bash
# Build first (see @ref building)
cmake --build build -t eulerSA -j8

# Run
mpirun -np 4 ./build/app/Euler/eulerSA cases/naca0012/eulerSA_config.json
```

> **TODO:** Add notes on:
> - Choosing the correct executable (`euler`, `euler3D`, `eulerSA`, `euler2EQ`, ...)
> - MPI rank count vs. mesh size guidelines
> - Setting `OMP_NUM_THREADS` for hybrid parallelism
> - Wall time and checkpoint restart

### Python Solver (EulerP)

> **TODO:** Add minimal Python script example using `DNDSR.EulerP`.

---

## Post-Processing

DNDSR outputs VTK (`.vtu` / `.pvtu`) and Tecplot (`.plt`) formats.

### ParaView / VisIt

- Open the `.pvtu` (parallel VTK) or `.vtu` (serial) files
- Available fields: density, velocity, pressure, temperature, Mach number,
  turbulence quantities (for RANS models)

> **TODO:** Add screenshots or a brief ParaView pipeline description.

### Tecplot

- Open `.plt` files directly
- > **TODO:** Any Tecplot-specific notes

---

## Tutorials

> **TODO:** Add step-by-step tutorials:
> 1. **Periodic Vortex** — 2D inviscid, validation case
> 2. **NACA 0012** — 2D RANS, airfoil at angle of attack
> 3. **3D Cylinder** — laminar / turbulent cylinder flow
> 4. **Custom case** — adapting an existing config to a new mesh

---

## Tips and Best Practices

> **TODO:** Collect practical advice:
> - Starting from a lower-order solution
> - CFL ramping for stiff cases
> - Monitoring residuals and convergence
> - When to use elevation + bisection vs. native O2 mesh
> - Memory usage estimates

---

## See Also

- @ref building — build instructions
- @ref solver_config — JSON configuration reference
- @ref python_geom_guide — mesh reading in Python
- @ref troubleshooting — common issues and fixes
