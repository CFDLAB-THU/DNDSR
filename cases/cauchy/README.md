# Native Cauchy cases

Run from the source root with `build/app/cauchy.exe cases/cauchy/<case>.json`.
Equation, EOS and transport selection live inside each config. Validate without
running with `--check-config`. These use the native DNDS/Geom/CFV solver and
SSPRK3 in the retained initial templates; they do not wrap legacy Euler.
Native implicit BDF1/2, Trapz, ESDIRK2/3/4 and DITR are available through
`time.method` and root `implicit` controls. The current implicit qualification
controls/results are kept separately in the independent plan repository at
`experiments/cauchy-implicit/`; the cancelled explicit viscous/DMR campaign is
not resumed merely by the continued presence of its template here.

| Case | Physical specification | Requested evidence |
| --- | --- | --- |
| `isentropic_vortex_20/40/80` | beta=5, periodic [0,10]^2, t=1, P3, no limiter or face fallback | Volume-average density L1/L2/Linf, mesh/time refinement, conservation |
| `cylinder_inviscid_subsonic` | Upper half, D=1, U=1, rho=1, M=.1 | Steady residual, Cp, pressure drag |
| `cylinder_supersonic_front` | Front half, D=1, U=1, rho=1, M=20, characteristic 3WBAP | Detached bow shock, stand-off, pressure |
| `cylinder_re20` | Full D=1, U=1, rho=1, mu=.05, M=.2, adiabatic no-slip | Steady Cd, wake, residual |
| `cylinder_re200` | Full D=1, U=1, rho=1, mu=.005, M=.2, small transverse wake seed | Transient through t=200, several late CL/CD cycles, St |
| `dmr_triangular_3wbap` | Mach-10, angle=60 deg, x0=1/6, UniformDM240U triangles, t=.25 | Fully active characteristic descending 3WBAP, shock/reflection morphology |
| `acm_plate_sa` | Coarse TMR plate, 273×193 points, Re=5×10⁶, P1 LSQ, pseudo-CFL cap 20 | Wall friction, layer profiles and nonlinear residual history |
| `acm_plate_sst` | Same plate and reconstruction, pseudo-CFL cap 100 | Wall friction, layer profiles and nonlinear residual history |
| `acm_translating_vortex_20` | Constant-density inviscid vortex, periodic [0,10]², t=10, P3, BDF1 | Volume-weighted velocity and gauge-aligned pressure errors, continuity RMS |

The ACM plate cases expect `data/mesh/flatplate_273x193.cgns`. Use the
273×193 PLOT3D grid from the [TMR flat-plate grids](https://tmbwg.github.io/turbmodels/flatplate.html)
and the plan repository's `experiments/cauchy-acm/convert_tmr_plate.py` to
create the CGNS mesh with `WALL`, `INLET`, `SLIP`, `TOP` and `OUTLET` patches.
Pass an alternative mesh location with `-k /domains/0/settings/mesh/file`.
Both plate inputs use the default ACM velocity-component rule plus the separate
continuity tolerance. SA uses GMRES(5) with ILU(2) and allows 50,000 nonlinear
corrections. SST allows 10,000. Both write volume and boundary snapshots every
2,500 corrections.
The last internal snapshot is useful for profile comparisons after a capped
stage. The solver publishes an accepted steady result only when the configured
tolerances pass.

For SUBOFF phase 1, run [run_cauchy_suboff_phase1.sh](../../scripts/run_cauchy_suboff_phase1.sh)
with the CUDA and AmgX build. Set `CUDA_VISIBLE_DEVICES` to select the
physical GPU. The script uses one unbound MPI rank, defaults to 16 setup
threads, and reads the existing SUBOFF mesh from the parent checkout.
`SUBOFF_MESH` and `CAUCHY_EXE` can override those paths.
Set `SUBOFF_CHECK_CONFIG=1` to print the resolved config.

The [SUBOFF case](acm_suboff_p1_gg_amgx.json) uses ACM–SST at Re=12 million
with direct P1 Green–Gauss and a single native AmgX V-cycle per update.
It allows 20,000 nonlinear updates, checkpoints every 2,000, and writes
volume and boundary visualizations every 4,000. History is written every
update. The final state and summary distinguish a capped solve from
a converged one. The launcher creates a fresh output directory and
duplicates stdout to its log.

The ACM vortex template uses a BDF1 start from the analytic cell averages.
For ESDIRK4, first project the time-zero field onto the discrete continuity
constraint using `cauchy_experiment_acm_vortex_projection`. Pass that checkpoint
with `--restart` and select ESDIRK4 through `/time/method`. The projection tool
accepts an optional alternative `IV10` Cartesian mesh path after its checkpoint
argument. Use the same grid for projection and the subsequent solve.

P3 means cubic reconstruction (nominal fourth-order spatial accuracy). The
vortex uses integrated cell-average initialization and analytic error, not
pointwise centroid values. `reconstructionTolerance` is the relative update
criterion; inspect `vr_relative_update` to detect an insufficient sweep cap.
Its time step also needs refinement before attributing the measured slope to
space alone.

The Mach20 and DMR cases enable `positivity` with `temperatureFloor=1e-10`.
The bound is on internal energy at Tmin, not on a substituted perfect-gas
gamma inside a Riemann solver. Reconstruction uses one mean-preserving beta
over face and volume quadrature points. SSPRK uses a shared conservative
high/low flux blend when needed; nonfinite data are errors, and an inadmissible
low-order candidate requires a smaller dt. History includes temperature minimum
and beta/alpha activity. The vortex cases keep PP disabled.

`CylinderInvisHalfA`: `wall` is the curved body, `WALL_INVIS` is the flat
symmetry plane. `CylinderB1` has a radius-.5 body and radius-200 farfield;
its smallest radial layer is about .002. The explicit diffusion and acoustic
stability constraints can make a one-hour run incomplete. A capped result is
not a steady solution or a verified vortex street. The compatibility default
`numerics.viscousGradient="primitive"` uses cell primitive gradients. Select
`"polynomial"` explicitly for cached polynomial derivatives and Euler's normal
face-jump construction; the [viscous operator contract](../../docs/architecture/CauchyImplicit.md#polynomial-viscous-operator)
describes filtered/unlimited selection and boundary behavior. P3 inviscid
reconstruction alone does not establish fourth-order viscous accuracy.

The supersonic front mesh includes exit faces at x=0 in `FAR`. The farfield
condition respects supersonic inflow/outflow and the subsonic incoming component
count, instead of clamping an outflow to the freestream.

Output is a compact accepted-state history, native persistent-ID restart,
and collective volume and boundary VTKHDF. Set `output.boundaryFacialFluxes`
to include numerical face fluxes. Histories and output
use separate reconstruction scratch so cadence does not affect the trajectory.
The C++ frontend uses DNDS MPI initialization/finalization and `argparse`:

```sh
# Invoke from the checkout root for these case-relative mesh paths.
mpirun -np 4 build-refactor/app/cauchy.exe cases/cauchy/isentropic_vortex_20.json \
  -k /time/method -v '"esdirk4"' -k /time/dt -v 0.01
```

`--check-config` prints the resolved configuration without mesh construction;
`--restart <file>` resumes accepted state/history. They are mutually exclusive.
Repeated `-k/--overwrite_key` and `-v/--overwrite_value` arguments pair in order,
using JSON pointers into the resolved configuration and JSON values. They apply
after profiles/defaults/domain settings, before final validation. For example,
`-k /domains/0/settings/execution/overlap -v false` overrides that domain's
file settings. Only existing resolved keys can be targeted; `--check-config`
shows their paths.
Quote string values as JSON, as above. Unknown or mistyped settings fail the
same validation as file input. MPI cleanup runs on help, config-only and error
paths as well as normal execution.

History uses `DNDS::CsvLog`: `output.historySegmentRows` defaults to 10000.
Segments are `<prefix>_history.csv`, `<prefix>_history_1.csv`, etc., each with
the same header. Readers must concatenate segments in numeric order. As with
the existing writer, rotation can create an empty trailing segment at an exact
row limit. Use a fresh output prefix for a resumed run to preserve old evidence.

The same executable accepts several domains, sharing ranks or using disjoint
rank groups. It selects the minimum stable/physical step across domains and
requires every domain's configured convergence condition for convergence termination.
Each domain keeps its own history, named `<prefix>_domainN_history.csv` (plus
segments), and volume/surface output, named `<prefix>_final_domainN...`.
Domain ordinals follow the configuration; their names and rank lists are
recorded in the resolved JSON. The single-domain filenames remain unchanged.

Restart permits changed MPI ownership/order and polynomial degree. It omits
reconstruction by default; missing or incompatible coefficients cold-start at
zero. Set `output.checkpointReconstruction=true` for compatible finite-sweep
continuation. BDF2 includes its accepted previous state/timestep. File metadata
and physical mesh records validate interpretation before accepted publication.
For multiple domains, `<prefix>_final.h5` is an HDF5 manifest. Its relative child
paths refer to native archives below `<prefix>_final.h5.parts/<generation>/`.
Copy the manifest and its `.parts` directory together. A write publishes the
manifest after every domain succeeds; old generations are retained and may be
removed explicitly when their manifests/evidence are no longer needed. Reading
validates all children and reserves all state/reconstruction destinations before
changing any accepted state. Rigid-interface packets are rebuilt at the stored
time. This is coordinated application-level publication, without process-crash
recovery or a filesystem durability guarantee.
Body force is the integrated wall momentum flux; Cd/Cl use `.5*rho*U^2*D`.
For the upper half body, pressure drag must be doubled for a full-body Cd.

For reactive transient runs, `time.sourceSplitting` defaults to `"coupled"`.
Set it to `"strang"` for Euler's constant-volume CVODE half-step, source-free
transport step, then another source half-step. Controls are `time.reactor`
(`rtol=1e-10`, `atol=1e-18`, `maxOrder=0`, `maxSteps=10000000`). Density,
momentum, total energy and RANS stay fixed during each source half-step. The
source duration is in code time; shared Physics applies the reference scales.
Reconstruction is retained, but multistep transport history resets across source
jumps as in Euler. This means split BDF2 takes its BDF1 startup equation each
step. Split diagnostics are flow-only; split steady mode is rejected. Use a
standalone SGS/LU-SGS/ILU solve when matching the Euler transient flame cases.

The dedicated campaign enforces aggregate <=256 ranks, OMP=1, one hour per
ordinary row and eight hours for DMR. A graceful wall/step cap returns 2;
process timeout returns 124. Both require further work before numerical
qualification. Acceptance records live in the independent plan repository at
`experiments/cauchy-p3/`.
