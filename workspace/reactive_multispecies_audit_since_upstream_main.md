# Reactive/Multispecies Audit Since `upstream/main`

Audit target: current `HEAD` diff against `upstream/main`.

Scope: additions and supporting changes for reactive and multispecies flow, including physical correctness, numerical correctness, control logic, multi-threading, MPI behavior, software robustness, geometry interaction, cases, and tests.

Additional geometry scope: current `src/Geom` module usability and robustness beyond only the changed diff, especially adjacency organization, local/global state discipline, distributed ghosting, interpolation, reorder, device-view assumptions, and Python binding exposure.

Key physical assumptions to preserve:

1. The gas model is ideal gas, but `cp`/`cv` can vary with temperature and composition, so temperature from conservative state requires an implicit thermodynamic solve.
2. Positivity preservation must subtract mixture base internal energy from plain Cantera internal energy: `rhoe_sensible = rhoE - rhoK - rhoe_base >= 0`, implying `T > TBase`.
3. Reactive chemistry should preserve total energy when chemistry changes composition; repair/clipping paths must adjust `rhoE` consistently if they alter `rhoe_base`.

## Status Summary

Open findings: 33

Withdrawn findings: 2 (`RMS-AUDIT-001`, `RMS-AUDIT-016`)

Fixed in working tree: 7 (`RMS-AUDIT-002`, `RMS-AUDIT-003`, `RMS-AUDIT-005`, `RMS-AUDIT-008`, `RMS-AUDIT-011`, `RMS-AUDIT-012`, `RMS-AUDIT-032`)

Severity counts:

| Severity | Count |
| --- | ---: |
| High | 0 |
| Medium | 27 |
| Low | 6 |

Category counts:

| Category | Count |
| --- | ---: |
| Reactive source control | 1 |
| Positivity / thermodynamics | 2 |
| Chemical Jacobian | 1 |
| Flux / multispecies numerics | 0 |
| Boundary / transport units | 0 |
| Cases / configuration / schema | 4 |
| Mesh / MPI geometry state | 16 |
| Geometry Python / device / usability | 5 |
| Geometry output robustness | 1 |
| Build / test robustness | 3 |

## Findings

### Reactive Source Control

#### RMS-AUDIT-001 — Withdrawn — Strang reactive source step does not update conservative state

Evidence: `src/Euler/EulerEvaluator.hxx:1804-1861`, `src/Euler/EulerSolver.hxx:1398-1401`, `src/Euler/EulerSolver.hxx:1490-1492`

Correction: `auto state = u[iCell]` is an Eigen view/map of the writable cell buffer, not a copy-out value. Writes to `state` update `u[iCell]` directly.

Impact: no current issue; the original finding was based on an incorrect copy assumption.

Suggested fix: none.

Status: Withdrawn / false positive.

#### RMS-AUDIT-002 — Medium — Tau source splitting is not identity when `reactiveSourceScale == 0`

Evidence: `src/Euler/EulerSolver.hxx:155-167`, `src/Euler/EulerEvaluator.hxx:1529-1532`

`sourceTauSplittingEnabled` depends on reactive-flow enablement but not on `reactiveSourceScale`. The point-implicit source path explicitly documents that the split form is ill-formed in the `S -> 0` limit and does not recover identity.

Impact: Setting `reactiveSourceScale = 0` for nonreactive debugging can still change species through the split residual rebuild path if tau splitting is enabled.

Suggested fix: disable tau splitting when `reactiveSourceScale == 0`, or reformulate the split source update so zero source and zero source Jacobian produce an exact identity update.

Status: Fixed in working tree. `src/Euler/EulerSolver.hxx:155-163` now separates requested vs active tau splitting and bypasses the split when `reactiveSourceScale <= 0`, with an explicit comment that the split is not identity-preserving in the `S -> 0` limit.

#### RMS-AUDIT-015 — Low — Point-implicit source update accepts warm cache but does not refresh it

Evidence: `src/Euler/EulerSolver.hxx:169-175`, `src/Euler/EulerEvaluator.hxx:1377-1385`, `src/Euler/EulerEvaluator.hxx:1501-1527`, `src/Euler/EulerEvaluator.hxx:1564-1565`, `src/Euler/EulerEvaluator.hxx:1695-1696`

Solver comments state that `cellTWarm` is refreshed by `PointImplicitSourceUpdate()`, but the function accepts the optional cache and never uses or writes it. Internal `EvaluateCellSource()` calls omit the warm-cache argument.

Impact: After point-implicit reactive species updates, later thermodynamic solves can use stale per-cell temperature guesses. This is mostly a robustness/performance issue, but stiff cases with large composition changes may see extra UV-solve iterations or convergence failures.

Suggested fix: pass `cellTWarm` through internal source evaluations where appropriate, and after accepting `uNew[iCell]`, recompute and store `phys_.temperature(uNew[iCell], oldGuess)` in the cache.

Status: Open.

### Positivity / Thermodynamics

#### RMS-AUDIT-003 — High — Species clipping after increments can invalidate sensible energy positivity

Evidence: `src/Euler/EulerEvaluator.hpp:1948-1970`, comparison repair path at `src/Euler/EulerEvaluator.hxx:1420-1442`

`AddFixedIncrement()` first uses `CompressInc()` to enforce density and thermal positivity, then clips/rescales transported `rhoY_k`. That composition repair changes `rhoe_base`, but the code does not adjust `rhoE` by the before/after base-energy change and does not re-check `rhoe_sensible >= 0` or `T >= TBase`.

Impact: A state can pass the positivity limiter, then become thermodynamically invalid after species clipping. This directly risks violating the intended `rhoe_sensible = rhoE - rhoK - rhoe_base >= 0` invariant.

Suggested fix: apply the same base-energy compensation pattern used by `PointImplicitSourceUpdate()` when clipping species, then revalidate temperature and pressure. Prefer limiting species increments before energy compression so final state repair is conservative and physically explicit.

Status: Fixed in working tree. Implemented decay-based repair in `CompressInc()` (`src/Euler/EulerEvaluator.hpp:1806-1952`): the accepted increment is progressively reduced over primary conservative and species variables while leaving the RANS gap unchanged, then the clipped candidate must satisfy the sensible-energy floor. `CompressInc()` returns the full clipped repaired increment `candidate - u`.

#### RMS-AUDIT-004 — Medium — Low-level mass-fraction repair can hide invalid conservative states

Evidence: `src/Euler/Chemistry/ChemicalSource.cpp:1043-1067`, callers in `src/Euler/Physics/PhysicsProperties.hpp:294-305` and `src/Euler/SourceTermContributor.hpp:430-456`

`ChemicalSource::massFractions()` clamps negative species and renormalizes before EOS/source evaluation. This makes Cantera calls robust, but it also evaluates thermodynamics, source terms, and Jacobians on a repaired composition while the conservative state may still contain invalid `rhoY_k` or `sum(rhoY_k) > rho`.

Impact: Invalid states can be masked until later; source/Jacobian consistency only holds for already-valid conservative states. This can make nonlinear failures nonlocal and difficult to diagnose.

Suggested fix: separate strict conservative-state species extraction from Cantera-safe normalization. Use strict extraction for validity checks and Jacobian consistency, and reserve normalization for controlled recovery paths that also repair energy/species explicitly.

Status: Open.

#### RMS-AUDIT-010 — Medium — 0D implicit reactor path does not enforce dependent-species simplex

Evidence: `src/Euler/Physics/ConstVolTrajectory.hpp:72-79`, `src/Euler/Physics/ConstVolTrajectory.hpp:163-165`

`stepImplicit()` clips only transported species lower bounds after Newton and does not enforce `sum(rhoY_k) <= rho` or repair the dependent species in the conservative state. Later sampling/evaluation calls `massFractions()`, which clamps and renormalizes for reporting.

Impact: The 0D validation path can advance and report a normalized composition that is not represented by its conservative state. This can hide negative dependent species or `sum(rhoY) > rho` after stiff or large-`dt` updates.

Suggested fix: after every 0D Newton update, enforce the independent/dependent species simplex on `U` itself and explicitly choose whether to preserve total energy or adjust base/sensible energy. Add a large-`dt` stiff regression that asserts `rhoY >= 0`, `sum(rhoY_k) <= rho`, and reported `Y` matches `U`.

Status: Open.

### Chemical Jacobian

#### RMS-AUDIT-008 — High — Chemical source Jacobian assumes species immediately follow energy

Evidence: `src/Euler/Chemistry/ChemicalSource.cpp:826-871`, `src/Euler/Chemistry/ChemicalSource.hpp:237-240`, `src/Euler/SourceTermContributor.hpp:423-425`, `src/Euler/SourceTermContributor.hpp:469-490`

`productionRatesAndJacobian()` hard-codes `speciesCol0 = iEnergy + 1` and writes species derivatives into `speciesCol0 + k`. Extended reactive states compute `Isp = nVars - Ns1`, allowing RANS variables between energy and species. The caller passes only `iEnergy`, then copies all returned columns into the solver Jacobian.

Impact: For reactive `NS_EX` cases with RANS variables between `rhoE` and `rhoY`, species derivatives are placed in RANS/passive columns while true species columns are wrong. Stiff implicit reactive solves can converge poorly or linearize the wrong system.

Suggested fix: pass `speciesCol0`/`Isp` into `productionRatesAndJacobian()`, or infer it as `dOmegadU.cols - (Ns - 1)`. Update the public contract to remove the `iEnergy + 1` assumption. Add a finite-difference Jacobian test with RANS variables present.

Status: Fixed in working tree. `ChemicalSource::productionRatesAndJacobian()` now infers the species column start as the last `Ns-1` columns (`src/Euler/Chemistry/ChemicalSource.cpp:826-828`), and the API comment was updated in `src/Euler/Chemistry/ChemicalSource.hpp:237-244`.

#### RMS-AUDIT-009 — Medium — Chemical source Jacobian does not match temperature-floor-clamped RHS

Evidence: `src/Euler/SourceTermContributor.hpp:440-445`, `src/Euler/Chemistry/ChemicalSource.cpp:883-924`, `src/Euler/Physics/ConstVolTrajectory.hpp:132-145`

The chemical source RHS evaluates Cantera at `Tcantera = max(Tphys, baseTemperature)` and recomputes pressure at the floor when clamped. The Jacobian path still applies the full unclamped thermodynamic chain rule, including nonzero `dT/d(rhoE)`, momentum, and density couplings.

Impact: Near positivity recovery states below the chemistry floor, the source RHS is a clamped function but the implicit Jacobian is for a different function. Newton/source-implicit updates can get wrong energy and momentum sensitivities where robustness is most needed.

Suggested fix: when the floor is active, assemble the Jacobian of the clamped RHS: zero temperature-derivative columns caused solely by `T(U)`, zero momentum/rhoE couplings through `T`, and retain only valid density/composition effects at fixed `Tfloor`. Add finite-difference checks below `TBase`.

Status: Open, pending discussion. A linearization that ignores clamp-boundary derivatives may be acceptable for robustness near invalid states; the remaining question is whether the Jacobian should simply approximate the smooth interior function or explicitly model the clamped source.

### Flux / Multispecies Numerics

#### RMS-AUDIT-032 — High — Roe acoustic decomposition uses total-energy jump instead of sensible/base-adjusted jump

Evidence: `src/Euler/Gas.hpp:1191`, `src/Euler/Gas.hpp:1541`, default `rsType = Gas::Roe` at `src/Euler/EulerEvaluatorSettings.hpp:374`

Roe-type flux is the default. Roe dissipation builds `incU` from raw conservative variables and uses the total-energy jump directly in acoustic wave strengths without subtracting the multispecies base-energy jump. The batched Roe path follows the same pattern.

Impact: A pure composition contact with equal `rho`, velocity, and pressure but different species base energies produces a total-energy jump. Roe interprets that as an acoustic/pressure jump, adding spurious momentum/energy dissipation and coupling species contacts into pressure waves. This can directly affect flame and detonation interfaces under the default solver.

Suggested fix: in Roe acoustic decomposition, use `Delta E_sensible = Delta E_total - Delta rhoE_base` for acoustic alphas while preserving total energy in conservative fluxes. Apply to scalar, batched, and implicit Roe-dissipation helper paths. Add a two-state multispecies contact test with same `rho/u/p` and different valid `Y`.

Status: Fixed in working tree. Scalar and batched Roe acoustic decomposition now subtract the base-energy jump from the energy jump used in acoustic wave strengths (`src/Euler/Gas.hpp:1198`, `src/Euler/Gas.hpp:1548-1556`).

### Boundary Conditions / Units

#### RMS-AUDIT-005 — High — `BCInPsTs` total pressure uses gas-constant scaling instead of pressure scaling

Evidence: `src/Euler/EulerBC.hpp:616-620`, pressure scale helpers in `src/Euler/Physics/PhysicsProperties.hpp:159-202`

`ResolveStateValues()` converts `BCInPsTs` total pressure with `phys.toCode()`, which scales gas constants and heat capacities by `R0 = U0^2/T0`. Pressure should be scaled by `p0 = rho0*U0^2`. Temperature conversion with `toCodeT()` is appropriate.

Impact: Non-unit reference scales impose the wrong stagnation pressure at total-condition inflow. Reactive runs then feed wrong static pressure into thermodynamic conversion and Cantera-dependent source/transport paths.

Suggested fix: add `toCodeP(pPhys) = pPhys / p0()` or divide by `phys.p0()` at this site. Add a BC test with non-unit `rho0`, `U0`, and `T0`.

Status: Fixed in working tree. Added `PhysicsProperties::toCodeP()` at `src/Euler/Physics/PhysicsProperties.hpp:197-198` and use it for `BCInPsTs` total pressure in `src/Euler/EulerBC.hpp:616-620`.

#### RMS-AUDIT-011 — Medium — Non-Cantera viscosity paths mix physical and code units

Evidence: `src/Euler/Physics/PhysicsProperties.hpp:364`, `src/Euler/Physics/PhysicsProperties.hpp:1484-1506`, `src/Euler/EulerEvaluatorSettings.hpp:618-639`, `src/Euler/SourceTermContributor.hpp:598-609`

`muGas` is documented as physical Pa*s and `printInfo()` reports `Code muGas = muGas / mu0`, but `muRef()` and non-reactive `mixtureViscosity()` return raw `igProp_->muGas`. Extended RANS source contributors also receive raw `settings.idealGasProperty.muGas`.

Impact: With non-unit `rho0`, `U0`, or `L0`, viscous and RANS terms can mix physical and code-scaled viscosity. Cantera laminar transport is scaled correctly, but fallback ideal-gas and RANS/turbulent paths can get wrong Reynolds/source scaling.

Suggested fix: make `muRef()` and non-Cantera viscosity paths return `muGas / mu0()`, and pass code-scaled viscosity into RANS contributors or rename/configure the field explicitly as code-scaled. Add a nondimensional viscous-flux test under changed reference scales.

Status: Fixed in working tree. `muRef()` and non-Cantera `mixtureViscosity()` now return code-scaled viscosity, and RANS source contributors receive code-scaled `muGas`.

### Cases / Configuration / Schema

#### RMS-AUDIT-012 — Medium — `DNDS_MECH_PATH` is prepended to project-relative mechanism paths

Evidence: `src/Euler/SourceTermContributor.hpp:629-635`, config validation path at `src/Euler/EulerSolver.hpp:1183-1200`, examples such as `cases/eulerEX/config_1d_detonation.json:176-178` and similar 2D detonation cases

Any non-absolute `mechanismFile` is prefixed with `DNDS_MECH_PATH`. Several committed cases use project-relative paths like `../cases/eulerEX/h2o2.yaml`, while notes also recommend setting `DNDS_MECH_PATH` to Cantera data. This resolves to `<DNDS_MECH_PATH>/../cases/eulerEX/h2o2.yaml`, not the project file.

Impact: Reactive example cases can fail mechanism validation/loading depending on environment, despite the configured path being valid relative to the build working directory.

Suggested fix: only prepend `DNDS_MECH_PATH` for bare filenames, or implement clear precedence: absolute path, working-directory/project-relative path, then `DNDS_MECH_PATH`/Cantera search. Normalize committed cases to one convention and test both path modes.

Status: Fixed in working tree. Mechanism resolution now uses the configured path if it exists, otherwise falls back through `DNDS_MECH_PATH`, otherwise leaves bare filenames to Cantera's data search.

#### RMS-AUDIT-013 — Low — Non-EX schemas advertise reactive flow settings that runtime rejects

Evidence: `cases/eulerSA_schema.json:2108-2164`, `src/Euler/EulerEvaluatorSettings.hpp:865-866`, `workspace/react_test_dense_50us.json:1-3`, `workspace/react_test_dense_50us.json:93-96`

Non-EX schemas such as `eulerSA_schema.json` expose full `reactiveFlow` settings, but runtime finalization rejects `reactiveFlow.enabled` unless the model is `NS_EX` or `NS_EX_3D`. A workspace reactive config points at an SA schema while enabling reactive flow.

Impact: Schema validation and editor autocomplete can bless configurations that runtime rejects, misleading users writing reactive/multispecies cases.

Suggested fix: generate model-aware schemas that omit `reactiveFlow` for non-EX models or constrain `enabled` to `false`. Ensure reactive examples reference `eulerEX_schema.json` or `eulerEX3D_schema.json`.

Status: Open.

#### RMS-AUDIT-022 — Low — Distributed HDF5 repartition ignores most configured partition options

Evidence: `src/Geom/Mesh/Mesh_ReadSerializeDistributed.cpp:340-354`, options defined at `src/Geom/Mesh/Mesh.hpp:1117-1122`

Distributed-read repartition ignores `metisType`, `metisUfactor`, `edgeWeightMethod`, and `metisNcuts`; it hard-codes `ubVec{1.05}` and `wgtflag{0}`, and only passes `metisSeed`.

Impact: Options that affect serial CGNS partitioning silently do nothing for redistributed HDF5 reads, making partition quality and reproducibility difficult to control.

Suggested fix: plumb supported options into ParMETIS or reject/document unsupported fields on this path. Add tests showing changed options affect ParMETIS inputs or emit explicit unsupported-option diagnostics.

Status: Open.

#### RMS-AUDIT-018 — Medium — Config validation rejects valid partitioned-mesh workflows

Evidence: `src/Euler/EulerSolver.hpp:1135-1166`, runtime read behavior at `src/Euler/EulerSolver_Init.hxx:61-83` and `src/Euler/EulerSolver_Init.hxx:181-197`

`validateConfigFiles()` always requires `dataIOControl.meshFile` to exist, even for `readMeshMode` values that read `meshFilePartitionedInput`. It also appends `.dnds.h5` or `.dir` unconditionally, while runtime input logic accepts and strips an already-present suffix.

Impact: Valid partitioned-mesh runs can fail before initialization when the original mesh is unavailable or `meshFilePartitionedInput` already includes `.dnds.h5`.

Suggested fix: validate `meshFile` only for workflows that read it, and normalize partitioned input exactly as runtime `getPartitionedMeshInput()` does before checking existence. Add config-validation tests for partition-only workflows.

Status: Open.

#### RMS-AUDIT-033 — Medium — Advertised reactive chemistry controls are unwired

Evidence: declarations/schema text at `src/Euler/EulerEvaluatorSettings.hpp:684-686` and `src/Euler/EulerEvaluatorSettings.hpp:697-701`

`reactiveFlow.CFLScale`, `chemRelaxEps`, and `chemAbsTol` are documented as active stiff-chemistry controls, but the solver/source paths use `reactiveSourceScale` and `reactorStepSettings` instead. These fields appear to be declarations/schema text only.

Impact: Users can tune advertised controls with no effect on timestep limiting, source relaxation, or chemistry tolerances. Stiff reactive runs can appear configured while numerics remain unchanged.

Suggested fix: wire these fields into their intended runtime paths, or mark them reserved and reject non-default values during config finalization. Add a config-validation test that non-default values either affect runtime controls observably or fail clearly.

Status: Open.

### Mesh / MPI Geometry State

#### RMS-AUDIT-006 — Medium — `InterpolateEdge()` leaves edge adjacencies in global-index state

Evidence: `src/Geom/Mesh/Mesh.cpp:1609`, `src/Geom/Mesh/Mesh.cpp:1665-1688`, `src/Geom/Mesh/Mesh.cpp:1691-1701`

`InterpolateEdge()` calls `BuildGhostEdge()`, which wires edge adjacency mappings and marks `cell2edge`, `edge2node`, and `edge2cell` as global, setting `adjEdgeState = Adj_PointToGlobal`. It does not call `AdjGlobal2LocalEdge()` afterward. Consumers that expect local edge adjacency after interpolation can interpret global IDs as local IDs.

Impact: Multi-rank edge consumers, device views, or geometry algorithms can fail or silently use wrong topology after edge interpolation.

Suggested fix: either call `AdjGlobal2LocalEdge()` at the end of `InterpolateEdge()` or make the API contract explicitly global and guard all consumers. Add a multi-rank 3D edge interpolation test that checks edge adjacency local-state invariants and index ranges.

Status: Open.

#### RMS-AUDIT-007 — Medium — `ReorderLocalCells()` proceeds through known-corrupt face/edge adjacency states

Evidence: `src/Geom/Mesh/Mesh_Reorder.cpp:684-701`, `src/Geom/Mesh/Mesh_Reorder.cpp:727-736`, `src/Geom/Mesh/Mesh_Reorder.cpp:763-785`, `src/Geom/Mesh/Mesh_Reorder.cpp:810-821`, `src/Geom/Mesh/Mesh_Reorder.cpp:877-888`, `src/Geom/Mesh/Mesh_Reorder.cpp:899-908`

The comments state that calling `ReorderLocalCells()` after face or edge interpolation will silently corrupt `cell2face`, `face2cell`, `cell2edge`, and `edge2cell`. The implementation has partial facial/C2F/N2CB handling, but edge adjacency remains unsupported: it never calls edge local/global conversions, never remaps `edge2cell`, never transfers `cell2edge`/`cell2edgePbi` rows, and never rewires edge mappings.

Impact: Users or future pipeline changes can reorder after interpolation and corrupt topology without a hard failure.

Suggested fix: add release-active preconditions rejecting non-unknown facial or edge adjacency states unless complete remapping support is implemented. Add a test that calls `InterpolateFace()` then `ReorderLocalCells()` and expects a controlled throw.

Status: Open.

#### RMS-AUDIT-014 — Medium — Pulled ghost-cell `cell2face` rows can contain negative encoded face IDs

Evidence: `src/Geom/Mesh/Mesh.cpp:1291-1339`, `src/Geom/Mesh/Mesh.cpp:1371-1387`, `src/Geom/Mesh/AdjIndexInfo.hpp:141-165`

`BuildGhostFace()` builds the face ghost map from local `Cell -> Cell2Face -> Face` needs, wires `cell2face`, and converts it to local. Later `MatchFaceBoundary()` pulls full `cell2face` rows for ghost cells through `cell2node` ghost indexing and then calls `AdjGlobal2LocalC2F()`. `AdjIndexInfo::toLocal()` encodes missing target-map entries as `-1 - globalIndex`.

Impact: Ghost-cell `cell2face` rows can contain negative encoded face IDs for ghost-cell faces not adjacent to local cells. Any consumer over `NumCellProc()` or device/ghost reconstruction code that assumes local face IDs can dereference invalid negative indices. The same pattern needs checking before localizing edge ghost rows.

Suggested fix: before converting pulled ghost `cell2face` rows, expand the face ghost map to include all global faces in pulled ghost-cell rows, or trim/clear ghost rows and document that only owner rows are valid. Add assertions around consumers that require complete ghost rows.

Status: Open, pending reclassification. Negative encoded IDs after local conversion are expected for referenced entities that are not locally stored; this should be treated as a consumer-contract issue only if downstream code dereferences those entries as local IDs.

#### RMS-AUDIT-020 — Medium — Ghost DSL can include `UnInitIndex` as a requested ghost entity

Evidence: `src/Geom/Mesh/MeshConnectivity_Ghost.cpp:343-344`, `src/Geom/Mesh/MeshConnectivity_Ghost.cpp:364-375`, `src/Geom/Mesh/MeshConnectivity_Ghost.cpp:556-574`

`traverseHopImpl()` inserts every adjacency entry into `seen` without skipping `UnInitIndex`. `filterNonOwned()` treats negative values as ghosts, and the result can be exposed as active ghost indices. Fixed-width adjacencies such as `face2cell` or `bnd2cell` can legitimately contain `UnInitIndex` in unused slots.

Impact: A ghost chain traversing optional fixed-width slots can request invalid global index `-1`, leading to invalid ghost mappings or scratch pulls. This is a generic DSL robustness issue even if standard specs mostly avoid those hops today.

Suggested fix: skip `UnInitIndex` in `traverseHopImpl()` before inserting into `seen`, and assert/reject other negative values unless explicitly supported. Add a ghost-tree unit test with fixed-width adjacency containing `UnInitIndex`.

Status: Open.

#### RMS-AUDIT-021 — Medium — Redistributed HDF5 read aborts on over-decomposed small meshes

Evidence: `src/Geom/Mesh/Mesh_ReadSerializeDistributed.cpp:336-338`, public contract at `src/Geom/Mesh/Mesh.hpp:926-927`

`ReadSerializeAndDistribute()` is documented as reading a mesh into any number of MPI ranks, but `ReadDistributed_PartitionParMetis()` asserts every rank has at least one cell before calling ParMETIS. Even-split HDF5 reads produce zero local cells when `mpi.size > nCellGlobal`.

Impact: Small meshes, smoke tests, or over-decomposed CI runs abort during redistributed HDF5 read instead of producing valid empty ranks or a controlled diagnostic.

Suggested fix: support empty ranks through an active subcommunicator/fallback partitioner, or reject over-decomposition before ParMETIS with a clear user-facing error and documented limit. Test a 1-2 cell serialized mesh on 4 ranks.

Status: Open.

#### RMS-AUDIT-023 — Medium — Boundary mesh extraction creates orphan nodes from skipped periodic boundaries

Evidence: `src/Geom/Mesh/Mesh.cpp:1881-1885`, `src/Geom/Mesh/Mesh.cpp:1961-1975`

`ConstructBndMesh()` marks boundary-mesh nodes from all `bnd2node` rows, including periodic boundaries, but later skips periodic boundary elements when creating boundary-mesh cells.

Impact: Periodic-only or mixed periodic/non-periodic meshes can create orphan nodes in the extracted boundary mesh. This inflates global node counts and can confuse VTK/CGNS output or boundary-only consumers that assume every boundary-mesh node is referenced by a boundary cell.

Suggested fix: build `node2bndNodeGlobal` only from boundary rows that will become boundary-mesh cells, or include periodic boundary cells consistently with clear semantics. Test periodic-only and mixed periodic/wall meshes.

Status: Open.

#### RMS-AUDIT-024 — Medium — `IsO1()` and `IsO2()` return rank-local results after global reduction

Evidence: `src/Geom/Mesh/Mesh.cpp:291-316`, `src/Geom/Mesh/Mesh.cpp:319-344`

`IsO1()` and `IsO2()` compute `hasBadAll` with `MPI::Allreduce`, but return `hasBad == 0` instead of `hasBadAll == 0`.

Impact: If only some ranks hold higher/lower-order elements, ranks can make different control-flow decisions about elevation, bisection, or order handling, risking wrong mesh processing or MPI divergence.

Suggested fix: return `hasBadAll == 0` in both functions. Add a two-rank mixed-order mesh test and assert both ranks report the same result.

Status: Open.

#### RMS-AUDIT-025 — Medium — CGNS multi-zone node deduplication depends on connectivity traversal order

Evidence: `src/Geom/Mesh/Mesh_Serial_ReadFromCGNS.cpp:57-87`

`AssembleZoneNodes()` deduplicates zone-interface nodes only when the donor zone has already been visited. If a connectivity record points to an unvisited zone, it pushes that zone onto `zonesFront` but does not record the equivalence unless reciprocal connectivity is later processed.

Impact: CGNS files with one-sided zone connectivity can import duplicated interface nodes, breaking cross-zone cell adjacency, repartitioning, and ghost construction.

Suggested fix: collect all `(zone,node) <-> (donorZone,donorNode)` equivalences first using union-find, then assign compact node IDs after all connectivity records are processed. Add a two-zone one-sided `Abutting1to1` import test.

Status: Open.

#### RMS-AUDIT-026 — Medium — Wall distance is built before mesh transforms and not transformed afterward

Evidence: `src/Euler/EulerSolver_Init.hxx:251`, `src/Euler/EulerSolver_Init.hxx:275`, `src/Euler/EulerSolver_Init.hxx:298`, `src/Geom/Mesh/Mesh.hpp:988`, consumers at `src/CFV/VariationalReconstruction.hpp:617` and `src/CFV/VariationalReconstruction.hpp:629`

Solver initialization builds wall distance before `meshRotZ` and `meshScale` coordinate transforms. `UnstructuredMesh::TransformCoords()` rewrites only coordinates and does not rotate or scale `nodeWallDist`.

Impact: With `meshBuildWallDist=1` plus scale or rotation, solver-visible wall-distance vectors remain in the pre-transform frame. RANS or anisotropic reconstruction consumers can use wrong distances or directions.

Suggested fix: move `BuildNodeWallDist()` after all coordinate rectification or transform `nodeWallDist` consistently under rotation and scaling. Add a wall mesh test with `meshScale` and `meshRotZ`.

Status: Open.

#### RMS-AUDIT-034 — Medium — `ReorderEntities()` destroy mode leaves edge and periodic face state stale

Evidence: `src/Geom/Mesh/Mesh_Reorder.cpp:529-550`

`ReorderEntities()` handles `destroyKinds` only for `EntityKind::Face`. Edge adjacencies are registered/skipped for destruction (`cell2edge`, `edge2node`, `edge2cell`), but `EntityKind::Edge` is never destroyed. The new `cell2facePbi` is also not reset in the face-destroy block.

Impact: A caller using `ReorderEntities()` with `destroyKinds={EntityKind::Edge}` can leave built edge arrays alive but excluded from remap/relocation, so edge arrays silently retain stale indices after cell/node reorder. For `destroyKinds={Face}`, `cell2facePbi` can remain stale public state.

Suggested fix: extend the destroy block to reset all edge arrays, PBI arrays, `edgeElemInfo`, and set `adjEdgeState = Adj_Unknown`; also reset `cell2facePbi` when destroying faces. Add distributed reorder tests with built faces/edges and destruction modes.

Status: Open.

#### RMS-AUDIT-037 — Medium — `ReorderEntities()` face/edge reorders leave transferred arrays without reattached sons/transformers

Evidence: `src/Geom/Mesh/Mesh_Reorder.cpp:583-594`, `src/Geom/Mesh/Mesh_Reorder.cpp:610-640`, `src/Geom/Mesh/Mesh_Reorder.cpp:651-667`, supporting behavior in `src/DNDS/PermutationTransfer.hpp:334-336`

`PermutationTransfer::transferRows()` resets `pair.son` after transfer. `ReorderEntities()` recreates global mappings for reordered `Face` and `Edge`, but the reattach step handles only `Cell`, `Node`, and `Bnd`. There is no corresponding `Face` or `Edge` branch to recreate sons and reattach transformers for `face2node`, `face2cell`, `face2bnd`, `edge2node`, `edge2cell`, PBI arrays, and companion arrays.

Impact: Explicit `EntityKind::Face` or `EntityKind::Edge` reorders that preserve those arrays can leave valid fathers with null sons or stale transformers. Later `Size()`, ghost rebuild, device transfer, local/global conversion, or topology access can assert or use stale topology.

Suggested fix: extend the post-transfer reattach step to cover all reordered face- and edge-parallel arrays, including periodic companions and element-info arrays, or reject explicit face/edge reorder plans until this path is fully supported. Add distributed tests preserving built face/edge connectivity through reorder and checking all relevant sons/transformers and states.

Status: Open.

#### RMS-AUDIT-038 — Medium — `ReorderEntities()` omits pull sets needed to remap preserved face/edge adjacencies

Evidence: `src/Geom/Mesh/Mesh_Reorder.cpp:348-391`

The registry collects off-rank old globals for `Cell`, `Node`, `Bnd`, and some `Edge` targets, but omits several registered adjacency sources. It does not collect face-target globals from `cell2face`/`bnd2face`, node-target globals from `edge2node`, or cell-target globals from `edge2cell`.

Impact: If a distributed reorder remaps faces, nodes, or cells while preserving built face/edge topology, lookup resolution can fail for off-rank globals that appear only in omitted adjacencies. Non-destroy face/edge-preserving reorder paths can abort or depend accidentally on partition-local topology.

Suggested fix: add every registered preserved adjacency to pull-set collection by target kind, especially `cell2face`/`bnd2face` for face targets, `edge2node` for node targets, and `edge2cell` for cell targets. Add a multi-rank test with built faces/edges, global adjacencies, topology-preserving reorder, and lookup assertions enabled.

Status: Open.

#### RMS-AUDIT-039 — Medium — `ReorderEntities()` accepts local secondary adjacency states but remaps entries as globals

Evidence: public contract at `src/Geom/Mesh/Mesh.hpp:573-575`, state check at `src/Geom/Mesh/Mesh_Reorder.cpp:511-513`, registry construction at `src/Geom/Mesh/Mesh_Reorder.cpp:266-280`, remap/follow paths at `src/Geom/Mesh/Mesh_Reorder.cpp:57-63`, `src/Geom/Mesh/Mesh_Reorder.cpp:243-252`, and `src/Geom/Mesh/Mesh_Reorder.cpp:455-467`

`ReorderEntities()` requires all adjacencies to be global in its public contract, but the implementation checks only `adjPrimaryState`. It registers built face, edge, N2CB, and related adjacencies regardless of each tracked `idx.state()`, then treats all entries as old global IDs during follow/remap lookup.

Impact: A common built mesh can have primary adjacencies converted to global while secondary groups such as face/C2F/N2CB/edge remain local. `ReorderEntities()` can then resolve local appended indices as globals, corrupting follow maps and topology or aborting in lookup resolution.

Suggested fix: add release-active checks that every built registered adjacency is in `Adj_PointToGlobal` state before registry construction, or internally convert all built groups to global before remapping and restore intended states afterward. Add a test with built N2CB ghosts left local, primary converted global, and `ReorderEntities(Cell)` invoked.

Status: Open.

#### RMS-AUDIT-040 — Medium — Preserved periodic `cell2facePbi` is not relocated with `cell2face` rows

Evidence: periodic face interpolation at `src/Geom/Mesh/Mesh.cpp:1104-1107` and `src/Geom/Mesh/Mesh.cpp:1246-1250`, periodic face-row usage at `src/Geom/Mesh/Mesh.cpp:1375-1380`, registry list at `src/Geom/Mesh/Mesh_Reorder.cpp:299-310`, self-periodic orientation logic at `src/Geom/Mesh/Mesh.hpp:868-877`

`cell2facePbi` is built and used as a cell-indexed companion to `cell2face`, but `buildReorderRegistry()` registers other PBI arrays and omits `cell2facePbi`.

Impact: In non-destroy face-preserving reorders, `cell2face` rows move/remap while `cell2facePbi` rows remain in old cell order. Self-periodic face orientation logic can then read mismatched PBI bits.

Suggested fix: register `cell2facePbi` as a cell-parallel companion whenever periodic faces are preserved, transfer its rows with `cell2face`, and reattach/borrow its transformer. Add a periodic face-preserving reorder test that checks `CellIsFaceBack`/`CellFaceOther` before and after reorder.

Status: Open.

#### RMS-AUDIT-041 — Medium — Reordered optional node companion arrays are not reattached after transfer

Evidence: companion registration at `src/Geom/Mesh/Mesh_Reorder.cpp:315-318`, companion relocation at `src/Geom/Mesh/Mesh_Reorder.cpp:210-219`, transfer behavior in `src/DNDS/PermutationTransfer.hpp:82-89`, and node reattach branch at `src/Geom/Mesh/Mesh_Reorder.cpp:621-625`

`ReorderEntities()` registers `coordsElevDisp` and `nodeWallDist` as node companions and relocates companions for reordered kinds. `transferRows()` leaves pair `son` state reset/stale, but the node reattach branch only handles `coords` and `node2nodeOrig`.

Impact: A cell reorder defaults to also reordering nodes. If wall distance or elevation displacement is already built, those arrays are transferred and then left with null or stale ghost-side state. Later ghost-index access or communication on `nodeWallDist` or `coordsElevDisp` can assert or read invalid data.

Suggested fix: include all transferred node companions in the node reattach/transformer rebuild step, or explicitly destroy unsupported optional arrays during reorder. Add a test that builds wall distance/elevation displacement, runs a node-affecting reorder, and verifies companion arrays have valid sons/transformers and data.

Status: Open.

#### RMS-AUDIT-042 — Medium — Boundary-to-face side caches stay stale after boundary or face reorder

Evidence: cache declarations at `src/Geom/Mesh/Mesh.hpp:106-107`, default reorder expansion at `src/Geom/Mesh/Mesh_Reorder.cpp:421-428`, registry/companion list at `src/Geom/Mesh/Mesh_Reorder.cpp:291-318`, local-cache invalidation at `src/Geom/Mesh/Mesh_Reorder.cpp:669-678`, consumers at `src/Geom/Mesh/Mesh_WallDist.cpp:33-34` and `src/Geom/Mesh/Mesh.cpp:1985-1996`

`bnd2faceV` and `face2bndM` are local side caches for boundary/face mappings. `ReorderEntities()` can reorder boundaries by default when cells are explicitly reordered, and tracked `bnd2face`/`face2bnd` arrays are registered/remapped/relocated. The side caches are neither registered as companions nor cleared in the local-cache invalidation block.

Impact: Later wall-distance building or boundary-mesh extraction can use stale local face indices and wrong boundary rows after a boundary or face reorder.

Suggested fix: rebuild or clear `bnd2faceV` and `face2bndM` after any boundary or face reorder, or replace consumers with tracked `bnd2face`/`face2bnd` arrays after converting to the required local state. Add a reorder test that builds these caches, reorders cells/boundaries/faces, and verifies cache invalidation or correctness.

Status: Open.

### Geometry Python / Device / Usability

#### RMS-AUDIT-027 — Medium — Python `prepare_mesh()` passes `{}` where C++ expects `WallDistOptions`

Evidence: `python/DNDSR/Geom/utils.py:289`, `python/DNDSR/Geom/utils.py:319`, `src/Geom/Mesh/Mesh_bind.hpp:186`

`prepare_mesh()` documents `wall_dist_options` but defaults to `{}` when a wall-distance predicate is supplied. The pybind binding expects `UnstructuredMesh::WallDistOptions`, not a Python dict.

Impact: `prepare_mesh(mesh, reader, wall_dist_predicate=...)` fails in the default options path, making the recommended Python helper unusable for wall-distance computation unless callers pass the exact bound options object.

Suggested fix: construct `Geom.UnstructuredMesh.WallDistOptions()` when options are `None`, and explicitly map dict keys if dict input is intended. Add a Python test for the default predicate path.

Status: Open.

#### RMS-AUDIT-028 — Medium — Python exposes edge arrays but not edge construction APIs

Evidence: `src/Geom/Mesh/Mesh_bind.hpp:72`, `src/Geom/Mesh/Mesh_bind.hpp:96`, `src/Geom/Mesh/Mesh.hpp:515`

Python bindings expose `cell2edge`, `edge2cell`, and `edge2node`, but only bind `InterpolateFace()`. C++ declares `InterpolateEdge()`, `BuildGhostEdge()`, and edge local/global conversions, but those are not bound.

Impact: Python users can inspect edge fields but cannot build them, making new edge geometry features effectively unusable from Python and blocking Python-side EulerP/edge-topology experiments.

Suggested fix: bind `InterpolateEdge()`, `AdjGlobal2LocalEdge()`, and `AdjLocal2GlobalEdge()`, and preferably expose a high-level helper option for edge construction. Add a small 3D Python edge-connectivity test.

Status: Open.

#### RMS-AUDIT-029 — Medium — Mesh device transfer omits wall distance

Evidence: `src/Geom/Mesh/Mesh.hpp:200`, `src/Geom/Mesh/Mesh.hpp:1055`, `src/Geom/Mesh/Mesh_DeviceView.hpp:504`

`UnstructuredMesh` owns `nodeWallDist`, but `op_on_device_arrays()` excludes it. `UnstructuredMeshDeviceView` has wall-distance accessors commented out.

Impact: `mesh.to_device()` does not transfer wall distance, and device-side consumers cannot query it. EulerP/CFV GPU paths cannot safely use wall-distance-based reconstruction or turbulence features.

Suggested fix: include `nodeWallDist` in device transfer lists and add guarded device-view accessors that assert the field is built. Add a host/device wall-distance consistency test.

Status: Open.

#### RMS-AUDIT-030 — Medium — Wall-distance builder lacks pipeline and method validation

Evidence: `src/Geom/Mesh/Mesh_WallDist.cpp:33`, `src/Geom/Mesh/Mesh_WallDist.cpp:45`, `src/Geom/Mesh/Mesh.hpp:1042`

`BuildNodeWallDist()` immediately indexes `bnd2faceV[iBnd]` and uses face arrays without checking that face interpolation and boundary matching have run. The options schema allows any `method >= 0`, but implementation handles only `0`, `1`, and `20`; unsupported methods can silently produce no triangles and huge distances.

Impact: Python or solver callers can crash or get poisoned wall distances for wrong pipeline order or typoed method values.

Suggested fix: add release-active checks for facial/C2F state and `bnd2faceV.size() == NumBnd()`. Validate `method` against supported values or use an enum. Add tests for calling before face interpolation and invalid method.

Status: Open.

#### RMS-AUDIT-035 — Low — Python `CellFaceOther`/`CellIsFaceBack` lost common two-argument usability

Evidence: `src/Geom/Mesh/Mesh_bind.hpp:171-175`

Python exposes only `CellFaceOther(iCell, iFace, ic2f)` and `CellIsFaceBack(iCell, iFace, ic2f)`. There is no compatibility overload/default for common non-self-periodic meshes where `ic2f` is not semantically needed.

Impact: Existing Python geometry scripts using the previous two-argument helpers fail at call time, even on non-periodic meshes where the extra slot can be inferred.

Suggested fix: add Python overloads or default `ic2f=-1`; accept two-argument calls when the face is not self-periodic and throw a clear message requiring `ic2f` only for self-periodic faces. Add binding tests for non-periodic and self-periodic cases.

Status: Open.

### Geometry Output Robustness

#### RMS-AUDIT-031 — Low — `BuildVTKConnectivity()` appends on repeated calls

Evidence: `src/Geom/Mesh/Mesh_Serial_BuildCell2Cell.cpp:637`

`BuildVTKConnectivity()` resizes offsets and types but never clears `vtkCell2node`; it reserves then appends with `push_back()`.

Impact: Calling `BuildVTKConnectivity()` twice appends duplicate connectivity while offsets describe only the latest build, corrupting later VTK/VTKHDF output. This is easy to hit from Python notebooks or repeated prepare/output workflows.

Suggested fix: clear `vtkCell2node` at function entry alongside offsets and types. Add an idempotency test that calls the builder twice and checks sizes/output remain unchanged.

Status: Open.

### Build / Test Robustness

#### RMS-AUDIT-016 — Withdrawn — `canteraConstVolTrajectory` uses `Cantera::appdelete()` without including Cantera API

Evidence: `app/Euler/canteraConstVolTrajectory.cpp:1-4`, `app/Euler/canteraConstVolTrajectory.cpp:190`, `app/Euler/canteraConstVolTrajectory.cpp:196`, target added at `cmake/DndsApps.cmake:159-161`

The app calls `Cantera::appdelete()` but does not include a Cantera header. `ChemicalSource.hpp` intentionally hides Cantera symbols behind PIMPL, so it does not provide this declaration.

Correction: round 2 robustness audit found that `app/Euler/canteraConstVolTrajectory.cpp` includes `Euler/Physics/ConstVolTrajectory.hpp`, which includes `cantera/zerodim.h`. The auditor also reported successful builds of `canteraConstVolTrajectory`, `eulerState`, and `euler_test_chem_ode` on current `HEAD`.

Impact: No current correctness/build finding. A direct include or wrapper may still improve clarity, but this is not tracked as an open audit issue.

Status: Withdrawn / false positive.

#### RMS-AUDIT-017 — Medium — `DNDS_USE_CANTERA=OFF` is not a reliable build mode

Evidence: `cmake/DndsOptions.cmake:19`, `src/Euler/CMakeLists.txt:12-16`, `src/Euler/Chemistry/ChemicalSource.cpp:26-127`, `src/Euler/Chemistry/ChemicalSource.cpp:164-166`

`DNDS_USE_CANTERA` is user-facing, but `euler_library_chemical` is always built. With Cantera disabled, helper functions such as `normalizeTransportModel()` are not defined while constructors still call them.

Impact: `-DDNDS_USE_CANTERA=OFF` can fail to compile instead of producing a non-reactive build or a clear unsupported-mode error.

Suggested fix: remove the option, gate reactive/EX targets and tests behind it, or provide complete no-Cantera stubs with clear runtime errors for reactive paths.

Status: Open.

#### RMS-AUDIT-019 — Medium — C++ tests have no default timeout unless environment variable is set

Evidence: `cmake/DndsTests.cmake:10-17`, `test/cpp/CMakeLists.txt:61-67`

When `DNDS_TEST_TIMEOUT` is unset, `DNDS_TEST_SET_TIMEOUT` is `OFF`, so `_dnds_maybe_set_timeout()` does not set a CTest timeout. The documented C++ test commands do not pass `ctest --timeout`.

Impact: Hung MPI or C++ tests can block local or CI runs indefinitely unless every caller supplies a timeout externally.

Suggested fix: set a default per-test timeout, such as 120 seconds, and allow environment or CTest overrides. Verify timeout properties with `ctest -N -V`.

Status: Open.

#### RMS-AUDIT-036 — Low — Schema update script can clobber committed schemas on failure

Evidence: `cases/update_schemas.sh:25-26`

The schema update script redirects the `mpirun | grep` pipeline directly into the committed schema file. Shell redirection truncates the output file before the schema emitter succeeds; with `set -euo pipefail`, a failure exits after the file is already empty or partial.

Impact: A transient app/schema-generation failure can replace a valid schema with an empty or partial file, causing misleading diffs or broken editor/CI schema validation.

Suggested fix: write to a temporary file and atomically move it over the schema only after the pipeline succeeds; remove the temporary file on failure. Add a shell test with a fake failing executable that verifies the original schema remains unchanged.

Status: Open.

## Checked Areas Without Current Findings

1. Variable-`cp/cv` temperature recovery: `PhysicsProperties::temperature()` uses a Cantera UV solve for reactive mixtures (`src/Euler/Physics/PhysicsProperties.hpp:1562-1607`), so the core path is not using a constant-gamma explicit temperature formula.
2. Cantera pressure units in primary source/transport paths: source evaluation and transport calls generally use `toPhysP()` or direct SI pressure construction (`src/Euler/EulerEvaluator_EvaluateDt.hxx:1719`, `src/Euler/Physics/PhysicsProperties.hpp:1505-1532`, `src/Euler/Physics/PhysicsProperties.hpp:1619-1651`). The concrete pressure-unit bug currently recorded is `BCInPsTs` conversion.
3. Main OpenMP chemistry source path: source contributors use a per-thread `ChemicalSource` pool and check OpenMP thread index against pool size (`src/Euler/SourceTermContributor.hpp:364-375`, `src/Euler/SourceTermContributor.hpp:620-641`), so no direct shared-Cantera race is currently recorded.
4. Main Cantera laminar transport scaling returns physical viscosity/conductivity/diffusivity to code units in the reactive path (`src/Euler/Physics/PhysicsProperties.hpp:1504-1535`). The recorded transport-unit issue is the non-Cantera/fallback/RANS path.
5. Distributed H5 read repartition flow was reviewed in round 1 without a new concrete finding beyond the recorded adjacency-state issues.

## Convergence Log

### Round 0 — Seed Audit

Seeded from direct review plus two focused subagent reviews.

New findings: RMS-AUDIT-001 through RMS-AUDIT-007.

Disposition: continue with independent whole-diff review passes. Subagents should read this document first and report only new findings, duplicates with stronger evidence, or corrections to existing findings.

### Round 1 — Independent Whole-Diff Audit

Subagents: thermochemistry/numerics, flux/BC/cases, MPI/geometry/parallelism, robustness/tests/build.

New findings: RMS-AUDIT-008 through RMS-AUDIT-019.

Duplicate confirmations: RMS-AUDIT-005, RMS-AUDIT-006, and RMS-AUDIT-007 were independently reviewed and not repeated.

Disposition: continue with a dedicated whole-geometry audit, including unchanged `src/Geom` code paths, then run convergence passes on the updated finding set.

### Round 1G — Whole-Geometry Audit

Subagents: adjacency-state invariants, distributed mesh operations, Python/device/usability.

New findings: RMS-AUDIT-020 through RMS-AUDIT-031.

Strengthened findings: RMS-AUDIT-007 and RMS-AUDIT-014 gained edge/reorder and ghost-row details.

Disposition: continue convergence passes over the expanded finding set.

### Round 2 — Convergence Audit

Subagents: physics/numerics, geometry, robustness/build.

New findings: RMS-AUDIT-032 through RMS-AUDIT-036.

Corrections: RMS-AUDIT-016 withdrawn as a false positive after successful target build evidence.

Disposition: run one final convergence pass requiring only genuinely new, non-duplicate findings.

### Round 3 — Final Convergence Attempt

Subagents: two independent final convergence auditors.

New findings: RMS-AUDIT-037 and RMS-AUDIT-038.

Corrections: one duplicate report strengthened RMS-AUDIT-037 rather than adding a separate finding.

Disposition: because new findings were still discovered, run a final narrow pass over `ReorderEntities()` and remaining medium/high gaps.

### Round 4 — Reorder-Focused Convergence

Subagents: one narrow `ReorderEntities()` auditor and one final broad auditor.

New findings: RMS-AUDIT-039 and RMS-AUDIT-040.

Rejected finding: a reported shared-Cantera OpenMP race was not recorded because `PhysicsProperties::chem()` selects the per-thread `ChemicalSource` from the pool via `omp_get_thread_num()`, and checked thermodynamic/transport/Strang paths go through that accessor.

Disposition: run one final narrow convergence check over the expanded `ReorderEntities()` cluster.

### Round 5 — Final Narrow Reorder Check

Subagent: one `ReorderEntities()`-only auditor.

New findings: RMS-AUDIT-041.

Disposition: a post-041 sanity pass was run because this round still found a new issue.

### Round 6 — Post-041 Reorder Sanity Check

Subagent: one `ReorderEntities()`-only auditor.

New findings: RMS-AUDIT-042.

Disposition: stop automated audit rounds for this session. The new findings in rounds 3-6 are concentrated in `ReorderEntities()` variants; additional review should likely start by fixing or formally disabling unsupported reorder modes, then re-auditing that subsystem. Continuing to launch more agents before that design decision is likely to produce more variants of the same unsupported-mode problem rather than independent audit coverage.
