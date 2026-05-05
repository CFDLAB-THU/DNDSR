# Troubleshooting and FAQ {#troubleshooting}

**Status:** Skeleton — please add entries as issues are encountered and resolved.

---

## Build Issues

### CMake cannot find MPI

> **TODO:** Common causes and fixes (missing `mpicc`, `MPICH` vs. OpenMPI,
> `CMAKE_PREFIX_PATH`, environment modules).

### CGNS / HDF5 linking errors

> **TODO:** Notes on `cfd_externals` build, version compatibility, and
> `LD_LIBRARY_PATH`.

### CUDA build fails

> **TODO:** Common nvcc / host compiler version mismatches, architecture flags.

---

## Runtime Issues

### Segfault or abort in Python tests

**Symptom:** `pytest` crashes immediately or produces wrong results.

**Cause:** Stale pybind11 `.so` files. The installed Python modules load `.so`
files from `python/DNDSR/`, which are only updated by `cmake --install`.

**Fix:**
```bash
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py
```

Also see AGENTS.md for the full Python test workflow.

### MPI tests hang

> **TODO:** Notes on `pytest-timeout`, `mpirun` vs. `srun`, firewall issues,
> process binding.

### "Ghost mapping not built" assertion

> **TODO:** Usually means `BuildGhostPrimary` or `InterpolateFace` was skipped
> in the mesh pipeline.

---

## Python Issues

### `ModuleNotFoundError: No module named 'DNDSR'`

> **TODO:** Explain editable install (`pip install -e .`) vs. `PYTHONPATH`.

### Type stubs are out of date

> **TODO:** How to regenerate `.pyi` files with `pybind11-stubgen`.

---

## Numerical Issues

### Solution diverges immediately

> **TODO:** Checklist: CFL too high, bad initial condition, mesh quality,
> boundary condition mismatch.

### Negative density / pressure

> **TODO:** Limiter settings, positivity-preserving options, CFL reduction.

### Slow convergence

> **TODO:** Jacobian update frequency, preconditioner choice, CFL adaptivity.

---

## How to Get Help

- Check this FAQ first
- Review @ref building and @ref user_guide
- For bugs or feature requests, open an issue on GitHub

---

## See Also

- @ref building — build instructions
- @ref user_guide — running simulations
- @ref solver_config — configuration options
