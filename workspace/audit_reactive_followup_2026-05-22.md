# Reactive Flow Follow-Up Audit — 2026-05-22

Base reviewed: `upstream/main` merge-base `8448ab51789df3e81c2af8b41eb7933dc2399116`.
Branch head at review start: `7bbc4ea` (`dev/harry`).

## Findings and Disposition

| # | Severity | Area | Disposition |
|---|----------|------|-------------|
| 1 | High | Chemical source Jacobian fluid columns | Deferred by request. Keep `JAC_SKIP_FLUID` while testing; local TODO must state that it drops density, momentum, and energy/temperature coupling. |
| 2 | High | Scalar inviscid flux gamma | Fix: add templated two-gamma path (`gammaL`, `gammaR`, `gammaLm`, `gammaRm`) while preserving const-gamma path. |
| 3 | High | Batched inviscid flux gamma and formation | Fix: reactive batched path computes per-face `gammaL/gammaR` and per-face `rhoH_form_L/R`; const-gamma path keeps scalar inputs. |
| 4 | High | `BCInPsTs` formation/stagnation conversion | Deferred by request. Mark incomplete with TODO because total-condition inflow needs a more careful stagnation-to-static iteration. |
| 5 | High | `BCWallIsothermal && noRsOnWall` formation density | Fix directly by scaling formation enthalpy to the updated wall density. |
| 6 | High | Species clipping after PP compression | Fix: only when clipping/rescaling occurs, adjust total `rhoE` by `rhoH_after - rhoH_before` to preserve sensible energy. |
| 7 | Medium | Chemical Jacobian reference-energy derivative | Fix: include the derivative of the DNDSR-to-Cantera reference offset (`pV_ref + e_sens_ref`) in species-column `dT/d(rhoY_k)`. |
| 8 | Medium | Special BCs in reactive mode | Fix: ban special far-field BCs for reactive flow with a release-active check. |
| 9 | Medium | Cantera hard dependency | Accepted for development. No change. |
| 10 | Medium | CTest mechanism path | Fix: set `DNDS_MECH_PATH`/`CANTERA_DATA` for C++ Euler tests and CI C++ test step. |
| 11 | Low | `eulerState` CLI diagnostics | Fix pressure diagnostic and audit CLI consistency. |

## Verification Plan

1. Build targeted Euler unit tests: `cmake --build build -t euler_test_source_chemical euler_test_uv euler_test_chem_ode euler_test_gas_thermo -j8`.
2. Run targeted CTest cases with output: `ctest --test-dir build -R "^(euler_source_chemical|euler_uv|euler_chem_ode|euler_gas_thermo)$" --output-on-failure`.
3. Build `eulerState` target if available, then run a representative reactive conversion using `DNDS_MECH_PATH`.

## Final Audit Notes

- The finite-difference chemistry Jacobian test now routes temperature recovery through `PhysicsProperties::temperature()` and mass-fraction recovery through `ChemicalSource::massFractions()`, avoiding duplicated conversion logic in the test.
- The analytical chemistry Jacobian now includes the reference-energy and composition-energy derivatives needed by the active `PhysicsProperties` temperature bridge.  The nonzero-velocity rho-column kinetic derivative remains documented and is masked by `JAC_SKIP_FLUID` in the production path.
- Reactive inviscid flux paths now use side-specific and mean-state gammas, plus per-face formation enthalpy in the batched Roe path.  Constant-gamma paths continue to instantiate the default scalar path and pass `nullptr` for unused vectors.
- Built-in special far-field BCs are release-banned for reactive flow; `BCInPsTs` remains explicitly marked incomplete for a future stagnation-to-static iteration.
- Serial and MPI Euler CTests now receive `DNDS_MECH_PATH` and `CANTERA_DATA`; CI also sets these on the C++ test step.
