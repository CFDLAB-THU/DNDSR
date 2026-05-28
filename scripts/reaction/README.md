# Reaction Reference Scripts

## H2/O2 1D Premixed Flame

`h2o2_free_flame_mixture_averaged.py` generates Cantera freely propagating
premixed-flame references for DNDSR reactive-flow development.

These references intentionally use:

- `h2o2.yaml`
- `FreeFlame`
- `transport_model = "mixture-averaged"`
- `soret_enabled = False`
- `flux_gradient_basis = "mass"`

This omits multicomponent and Soret diffusion, matching the transport physics
currently targeted in DNDSR more closely than the full upstream Cantera example.

Reproduce the committed references with:

```bash
python scripts/reaction/h2o2_free_flame_mixture_averaged.py --loglevel 0
```

If Cantera cannot find `h2o2.yaml`, set `CANTERA_DATA` or pass an explicit path:

```bash
python scripts/reaction/h2o2_free_flame_mixture_averaged.py \
  --mechanism external/cfd_externals/repos/cantera/data/h2o2.yaml \
  --loglevel 0
```

Current Cantera 3.2.0 results:

| Case | Inlet composition | Flame speed |
| --- | --- | ---: |
| `cantera_h2_o2_ar` | `H2:1.1, O2:1, AR:5` mole fractions | `0.718742509 m/s` |
| `react_test_stoich_air` | `Y_H2=0.028, Y_O2=0.222, Y_N2=0.75` | `2.253965977 m/s` |

Outputs are written under `scripts/reaction/reference/`:

- `*_profile.csv`: spatial profiles with temperature, density, velocity, heat release, and all mass fractions.
- `*.yaml`: Cantera restore files for post-processing or restart.
- `h2o2_free_flame_mixture_averaged_summary.json`: machine-readable summary values.
