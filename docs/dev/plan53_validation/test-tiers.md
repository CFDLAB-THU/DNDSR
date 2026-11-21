# Plan 53: focused test tiers and MPI ranks

For the current cumulative CI levels, build targets and full catalogs, see the [test guide](../../tests/overview.md). The labels below document the original workload classifications.

Tier labels select a routine correctness gate without removing any executable,
case, assertion or MPI registration. Unfiltered CTest still runs all tiers.
`ctest --test-dir build-audit -L routine --output-on-failure` selects routine
coverage; `-L extended` selects additional MPI repetition and `-L integration`
selects the retained legacy mesh/golden fixtures. `mpi-high` is an independent
label describing np4/8, not a synonym for extended.

The following Cauchy families retain np1/2 in routine and np4/8 in extended:

| Suite | Why routine needs np2 | Why np4/8 is supplementary here |
| --- | --- | --- |
| boundary_inputs | Boundary point values with distributed cells | The input representation and boundary-state oracle do not introduce another rank role above two |
| wall_distance | Nodal halo interpolation and retained shape samples | More partitions repeat the same node-to-cell interpolation contract |
| rans_fv | Model source/transport, current-gradient halos and split layouts | Physical formulas and worker/inline operator agreement already execute at np1/2; higher-rank repetitions remain available |
| equation_source | Species/chemistry source and conservative Jacobian | Pointwise physical oracles plus the np2 source halo do not require four distinct owners |
| viscous_flux | Distributed viscous face contributions | Additional rank counts exercise more partitions of the same stencil |
| auxiliary_preparation | Prepared signals and stale/current input checks | Rank-asymmetric rejection already has a rejecting and a nonrejecting participant at np2 |
| nonlinear | Shared correction/acceptance orchestration | Local iteration mathematics and distributed acceptance already execute at np2 |
| physical_timestep | Global time bounds and rank-local invalid controls | np2 supplies distinct local states and collective reduction participants |
| implicit_domain | CPU implicit stage/step composition | Higher-rank runs repeat the same method/linear composition |
| foundation | Configuration, dispatch and driver wiring | Its two-rank orchestration is the routine distributed gate |
| polynomial_multigrid | Per-stage pFAS ownership and coarse/fine mathematics | Extra ranks supplement the same bounded hierarchy; these are not physical MG qualification cases |

Keep routine high-rank coverage for the remaining suites. In particular,
`equation_dofs` creates an empty owner only above two ranks and tests a
three-owner logical ring plus empty rank at np4. `finite_volume` and
`limited_vr_checkpoint` deliberately use empty owners above np2. `checkpoint`
uses rank-subset writers; np4 covers two writing ranks followed by four
reading ranks, which np2 cannot represent. Multidomain tests retain the
communicator-partition cases. Core Runtime/Geom route, graph, ghost and
redistribution suites retain their high-rank coverage. This is selective tier
curation rather than a blanket reduction to np2.

Existing serial configuration/physical-property tests remain serial. The
shared-library configuration fixture retains its separate DSO boundary.
Large 32^3 CFV golden/convergence cases remain in integration; their generated
64-cell routine contract remains available at all configured ranks. Expensive
mesh fixtures are lazy, and serial Python extension coverage has one CI owner.
All of these distinctions preserve test purposes rather than weakening
numerical thresholds or changing fixtures to force a passing result.

Timing/count observations belong in the execution ledger after configuration
and execution. Tier labels alone establish neither a runtime saving nor full
suite success. Matched performance benchmarks stay opt-in and outside CTest.
