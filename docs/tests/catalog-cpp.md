# C++ test catalog

Each heading is a build target. The table gives the minimum CI level of each CTest invocation. The source lists below it give the doctest case names in that binary. Rank, build features and doctest filters can restrict an invocation to part of its binary. Template cases appear once per definition. The list combines the configured profiles supplied to the generator.

## `cauchy_solver`

| Level | CTest registrations |
|---|---|
| 0 | `cauchy_frontend_np2` |
| 1 | `cauchy_frontend_np1` |

[Frontend test driver](../../test/Cauchy/frontend_smoke.py)

## `cfv_test_distributed_interface_sampling`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_distributed_interface_sampling_np2` |
| 1 | `cfv_distributed_interface_sampling_np1` |
| 2 | `cfv_distributed_interface_sampling_np4`, `cfv_distributed_interface_sampling_np8` |

[test_DistributedInterfaceSampling.cpp](../../test/cpp/CFV/test_DistributedInterfaceSampling.cpp)

- Distributed polynomial sampling evaluates on cell owners and returns query order
- Warped quad interface projects offset queries before distributed polynomial sampling
- Coupling participants without a donor mesh sample a reversed donor subgroup in query order

## `cfv_test_fv_facility`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_fv_facility_np2` |
| 1 | `cfv_fv_facility_np1` |
| 2 | `cfv_fv_facility_np4`, `cfv_fv_facility_np8` |

[test_FVFacility.cpp](../../test/cpp/CFV/test_FVFacility.cpp)

- FV metrics share managed mesh communication with independent preflight policy
- Midpoint face publication preserves integrated measures and cell metrics
- Runtime FV matches legacy VR sweeps and reproduces cell-average polynomials
- Independent functional weights and all-order boundary penalties match configured legacy sweeps
- WBAP explicit five-component face transforms match legacy and preserve a passive sixth field
- FV rejects wrong association and invalid shapes before writes
- Periodic quad and triangle reconstruction matches relaxed legacy Jacobi and ordered SOR
- Three-dimensional FV reproduces cubic cell-average polynomials on tetrahedron and hexahedron with empty ranks
- SVD removes unconstrained modes and invalid settings fail collectively
- Two-buffer SOR preserves its MPI input and split automatically selects core-first order

## `cfv_test_gradient_plan`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_gradient_plan_np2` |
| 1 | `cfv_gradient_plan_np1` |
| 2 | `cfv_gradient_plan_np4`, `cfv_gradient_plan_np8` |

[test_GradientPlan.cpp](../../test/cpp/CFV/test_GradientPlan.cpp)

- Standalone FV gradients recover all three spatial directions on a volume cell
- Standalone gradients recover linear fields with physical boundary samples
- Weighted P1 gradient matches independent least-squares fit on a stretched grid
- Weighted P1 reproduces a rotation-equivariant affine vector across a periodic seam
- Weighted P1 gradients use the curved triangular boundary sample locations
- Gradient core phases exclude active MPI ghost writes across table layouts
- Physical-gradient WBAP is direct, characteristic-optional, and split-equivalent

## `cfv_test_limited_vr`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_limited_vr_np2` |
| 1 | `cfv_limited_vr_np1` |
| 2 | `cfv_limited_vr_np4`, `cfv_limited_vr_np8` |

[test_LimitedVR.cpp](../../test/cpp/CFV/test_LimitedVR.cpp)

- Native v2/v3 gates preserve legacy thresholds, contact response and expansion admission
- Supported extremum seeds preserve incidence counts, retention and boundary exclusion
- LVR neighbor rounds cross rank boundaries with synchronous Cartesian distance and saturation oracles
- Distributed v2/v3 correction seeds distinguish mean contact support, transverse equality and history
- A-weighted limited VR preserves generic components, relaxation and split SOR semantics

## `cfv_test_limiters`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_limiters` |

[test_Limiters.cpp](../../test/cpp/CFV/test_Limiters.cpp)

- PolynomialSquaredNorm<2> with 2 rows (P1)
- PolynomialSquaredNorm<2> with 3 rows (P2)
- PolynomialSquaredNorm<2> with 4 rows (P3)
- PolynomialSquaredNorm<3> with 3 rows (P1)
- PolynomialSquaredNorm<3> with 6 rows (P2)
- PolynomialSquaredNorm<3> with 10 rows (P3)
- PolynomialDotProduct<2> self-product equals PolynomialSquaredNorm
- PolynomialDotProduct<2> linearity
- FMINMOD_Biway: same-sign inputs
- FMINMOD_Biway: opposite-sign inputs produce zero
- FMINMOD_Biway: zero input
- FVanLeer_Biway: same-sign inputs
- FVanLeer_Biway: opposite-sign produces zero
- FVanLeer_Biway: equal inputs return same value
- FWBAP_L2_Biway: identical inputs pass through
- FWBAP_L2_Biway: output bounded by inputs for same-sign
- FWBAP_L2_Cut_Biway: opposite-sign cut to zero
- FWBAP_L2_Multiway: all identical inputs pass through
- FWBAP_L2_Multiway: no NaN for random inputs
- FWBAP_L2_Multiway_Polynomial2D: all identical pass through
- FWBAP_L2_Multiway_Polynomial2D: no NaN for random inputs (nRows=2,3,4)
- FMEMM_Multiway_Polynomial2D: center input unchanged when smallest
- FMEMM_Multiway_Polynomial2D: no NaN for random inputs
- FWBAP_L2_Multiway_PolynomialOrth: all identical pass through
- FWBAP_L2_Multiway_PolynomialOrth: no NaN
- FWBAP_L2_Biway_PolynomialNorm<2,1>: identical inputs pass through
- FWBAP_L2_Biway_PolynomialNorm<2,Eigen::Dynamic>: no NaN
- FWBAP_L2_Biway_PolynomialNorm<3,Eigen::Dynamic>: no NaN for 3D dims
- FMEMM_Biway_PolynomialNorm<2,Eigen::Dynamic>: u2 smaller returns u2
- FMEMM_Biway_PolynomialNorm<2,Eigen::Dynamic>: no NaN
- FWBAP_L2_Biway_PolynomialOrth: identical inputs pass through
- FWBAP_L2_Biway_PolynomialOrth: no NaN
- Limiter ordering: FMINMOD output <= FWBAP output for same-sign
- All limiters handle near-zero inputs without NaN

## `cfv_test_reconstruction3d`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_reconstruction3d_contract_np2` |
| 2 | `cfv_reconstruction3d_contract_np1`, `cfv_reconstruction3d_contract_np4`, `cfv_reconstruction3d_contract_np8`, `cfv_reconstruction3d_np1`, `cfv_reconstruction3d_np2`, `cfv_reconstruction3d_np4`, `cfv_reconstruction3d_np8` |

[test_Reconstruction3D.cpp](../../test/cpp/CFV/test_Reconstruction3D.cpp)

- 3D: constant exactness with GaussGreen
- 3D static reconstruction mask is exchanged and survives GG and VR updates
- 3D: constant exactness with VFV_P1_HQM
- 3D: constant exactness with VFV_P2_HQM
- 3D: smooth-function golden values, convergence and order
- 3D: cell volumes sum to 1.0 (unit cube)
- Small connected 3D cells preserve constants through GG and P1/P2 VR

## `cfv_test_reconstruction_cuda`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_reconstruction_cuda_np2`, `cfv_reconstruction_nccl_np2` |
| 2 | `cfv_reconstruction_cuda_np1`, `cfv_reconstruction_nccl_np1` |

[test_ReconstructionCUDA.cu](../../test/cpp/CFV/test_ReconstructionCUDA.cu)

- Queued FV gradients and linear-mode projection share CPU and CUDA incidence semantics
- Queued polynomial sampling covers cached and direct derivatives on owned and ghost rows
- Shared quadrature CUDA placement reuses face allocation and retains queued scatter inputs
- Resident face basis supports direct and partial caches without unrelated metric payloads
- Geometry-only FV metrics can live on CUDA without retaining a host metric plan or allocating VR
- Device publications retain actual mesh metrics and basis with explicit optional capabilities
- Device-only reconstruction placement releases host payloads and remains executable
- CUDA sweep-only publication omits limiter matrices and scratch
- Raw VR packs B and the full inverse independently while using the FP64 sweep
- Opt-in FP32 sweep maps retain FP64 coefficients and avoid duplicate device matrix storage
- CUDA Jacobi and WBAP agree with CPU for physical DOF layouts and direct MPI split work
- CUDA preserves rotated periodic vectors and axisymmetric prepared operators
- CUDA degree three 3D reconstruction supports 37 coupled components and empty peers
- Device errors reject unsupported SOR stale publications and invalid indicators without silent output

[test_ReconstructionCUDA_AMG.hpp](../../test/cpp/CFV/test_ReconstructionCUDA_AMG.hpp)

- Single-device VR AMG preserves the projected modal matrix and fixed component corrections

[test_ReconstructionCUDA_Staged.hpp](../../test/cpp/CFV/test_ReconstructionCUDA_Staged.hpp)

- Stage-once VR pipelines preserve complete modal batches and queued dependencies
- CUDA VR launch policy rejects a non-warp thread count
- Float-float VR preserves low limbs through a cancelling modal product

[test_ReconstructionCUDA_Storage.hpp](../../test/cpp/CFV/test_ReconstructionCUDA_Storage.hpp)

- Contiguous VR payloads and raw face storage preserve oriented distributed sweeps
- Face cross blocks transpose fully while only symmetric inverse entries use triangular storage

## `cfv_test_reconstruction_math`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_reconstruction_math` |

[test_ReconstructionMath.cpp](../../test/cpp/CFV/test_ReconstructionMath.cpp)

- Compute scalar Taylor contexts preserve derivatives across explicit arithmetic choices
- Prepared Taylor points preserve nonunit-scale physical derivatives and zero-mean modes
- Taylor derivative pullback preserves Frobenius tensor norms through degree three
- Central isotropic weight schemes preserve configured legacy formulas

## `cfv_test_reconstruction_overlap`

| Level | CTest registrations |
|---|---|
| 0 | `cfv_reconstruction_overlap_np2`, `cfv_reconstruction_overlap_omp2_np2` |
| 1 | `cfv_reconstruction_overlap_np1` |
| 2 | `cfv_reconstruction_overlap_np4`, `cfv_reconstruction_overlap_np8` |

[test_ReconstructionOverlap.cpp](../../test/cpp/CFV/test_ReconstructionOverlap.cpp)

- Jacobi and WBAP core work overlaps real direct MPI and preserves unsplit results
- Core-first SOR uses a separate frozen send snapshot during split execution
- One-cell OpenMP SOR blocks reduce to Jacobi with frozen cross-block reads
- CFV split operators share eager ordering across inline and worker CPU launchers
- Queued CPU reconstruction and WBAP retain operands through chained publications
- CPU FV reconstruction limiter and linear actions agree across mixed DOF physical layouts

## `dnds_test_access_tracking`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_access_tracking` |

[test_AccessTracking.cpp](../../test/cpp/DNDS/test_AccessTracking.cpp)

- Checked prefixes reject missing stale and incomplete fan-in dependencies
- Backend rejection rolls back aliases without holding the ledger across work
- Explicit cross-context events prevent independent alias authority
- CPU worker checks submission before completion and preserves real event execution
- Checked release consumes aliases only after an explicit fan-in
- Immediate and unchecked boundaries reject pending conflicting checked accesses
- Inline checked execution publishes immediately and concurrent issue stays ordered

## `dnds_test_array`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_array` |

[test_Array.cpp](../../test/cpp/DNDS/test_Array.cpp)

- Ownership: host and Host-device leases retain fixed allocations
- Ownership: CSR row leases survive reserve compression and deletion
- Ownership: uncompressed row hashing preserves non-arithmetic values
- CSR sharing: public row starts and const iterators are read only
- Audit regression: compressed CSR hash includes row offsets
- Audit regression: host mirror publishes the current allocation
- Array TABLE_StaticFixed layout
- Array TABLE_Fixed layout
- Array TABLE_StaticMax layout
- Array TABLE_Max layout
- Array CSR layout
- Array clone and copy
- Array edge cases
- Array serialization round-trip
- Array signature
- Array hash
- Array resize-write-read

## `dnds_test_cauchy_boundary_inputs`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_boundary_inputs_np2` |
| 1 | `dnds_cauchy_boundary_inputs_np1` |
| 2 | `dnds_cauchy_boundary_inputs_np4`, `dnds_cauchy_boundary_inputs_np8` |

[test_BoundaryInputs.cpp](../../test/cpp/Cauchy/test_BoundaryInputs.cpp)

- Analytic boundary numerical failures return status through both equation entries
- DMR bottom producer resolves prescribed and reflecting pieces in world coordinates
- VR wall target selection preserves the selected physical boundary equation
- Total inlet resolves physical vector input and applies static density to SA
- Moving ACM interface packets preserve a uniform inertial flow under a common pressure shift
- P1 least squares uses affine receiver face-center values from an interface packet
- Published interface packets reproduce analytic P3 boundaries through auxiliary RHS and frozen Jacobian operators
- Physical and interface gradient targets evaluate nonconstant fields at actual face centers

## `dnds_test_cauchy_checkpoint`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_checkpoint_np4` |
| 1 | `dnds_cauchy_checkpoint_np1`, `dnds_cauchy_checkpoint_np2` |
| 2 | `dnds_cauchy_checkpoint_np8` |

[test_Checkpoint.cpp](../../test/cpp/Cauchy/test_Checkpoint.cpp)

- Accepted domain snapshots pin one immutable mean generation
- Prepared checkpoint owns its fields metadata and communicator beyond the domain
- Detached host snapshots retain accepted values after the domain mutates
- Asynchronous HDF output snapshots survive foreground field mutation and MPI
- Boundary snapshot writes native zones and suppresses viscous slip-wall flux
- Boundary snapshot follows the face owner when the boundary record is remote
- Equation archive remaps coupled and split fields without confusing component ranges
- Mixture restart redistributes every block and cold starts changed polynomial degree
- Mixture BDF2 restart restores species history and warm reconstruction transactionally
- Physical Strang source composes with transport and preserves retry and restart history semantics
- Mixture visualization shares volume species fields with adjacent boundary cells
- Configured RANS BDF2 restores all fields across layouts and rejects changed model metadata
- Older viscous archives retain primitive gradients and reject a silent polynomial method change
- Older physical-boundary archives omit inactive composition without invalidating restart
- Checkpoint changes MPI partition and polynomial degree using persistent cell IDs with cold reconstruction
- ACM BDF2 restart preserves the pressure gauge and differential history
- A mean-only initial checkpoint can seed a different nonlinear stage policy
- ACM internal restart adopts the stored absolute pressure target
- Implicit restart preserves variable BDF2 history and rebuilds live DITR stage bindings
- Restart dt growth reads the accepted scalar and supports files predating it
- CGNS checkpoint remains physical across independently built MPI partitions
- Rigid motion restart retains absolute-state convention and BDF history
- Means-only restart selects the steady trial or the accepted transient state
- Exact internal checkpoints resume BDF DIRK and DITR nonlinear iterations

## `dnds_test_cauchy_config`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_config` |

[test_Config.cpp](../../test/cpp/Cauchy/test_Config.cpp)

- Automatic CUDA policies select measured A100 kernels and preserve explicit overrides
- CUDA RHS launch selection respects measured profiles and explicit values
- Mesh preprocessing is explicit and validates a positive uniform scale
- Programmed initialization preserves order six by default and validates overrides
- Native AMG is selected independently for reconstruction and flow
- Diagonal inverse refinement is opt-in and rejects unrelated inverse providers
- Reconstruction kernel selection is explicit and retains the FP64 default
- Coarse solver resolution respects equation width and the selected matrix provider
- Linear matrix storage round-trips independently and rejects hidden AmgX repacking
- CUDA capability resolution requires a complete combined perfect gas profile
- CPU workers require an MPI capability that permits multithreaded execution
- Cauchy exposes one canonical layered input schema
- MPI fail-fast launch validation is an explicit execution choice
- DITR inherits common controls and only supplied compatibility fields override them
- Chemical source splitting is explicit and keeps Euler reactor defaults
- Configured RANS exposes Euler DES and QCR controls without a unit-length substitution
- Halo freshness audits are opt-in and the collective check switch overrides them
- Interface capability checks admit coordinated DITR and honor the steady BDF1 override
- Polynomial viscous reconstruction is explicit and retains the previous primitive default
- Physical timestep PP has explicit validated controls and is disabled by the steady limit
- Strict implicit VR acceptance needs an explicit positive reconstruction tolerance
- Steady config resolves Euler's BDF1 infinite timestep override with provenance
- Deprecated time and output aliases are exclusive with their canonical controls
- Reconstruction PP controls are explicit and reject unsupported compatibility combinations
- Cauchy implicit controls resolve through the root and reject contradictory paths
- Cauchy partial analytic boundary profile receives nested schema defaults
- Cauchy root resolves profiles overrides CLI and condition defaults with provenance
- Cauchy rejects unknown nested keys unused contradictory settings and unsupported capabilities

## `dnds_test_cauchy_config_shared`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_config_shared` |

[test_ConfigShared.cpp](../../test/cpp/Cauchy/test_ConfigShared.cpp)

- Hidden consumers serialize complete configuration registered in the shared library

## `dnds_test_cauchy_domain_cuda`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_domain_cuda_np2` |
| 2 | `dnds_cauchy_domain_cuda` |

[test_DomainCUDA.cpp](../../test/cpp/Cauchy/test_DomainCUDA.cpp)

- ACM device correction matches CPU on flow4 and mechanical SA and SST
- ACM CUDA exports zero tangential slip-wall flux and nonzero no-slip shear
- ACM polynomial CUDA face mappings match the host on a rotational seam
- Automatic A100 kernels advance raw P3 flow with frozen explicit reference selectors
- Device domains advance nonuniform implicit flow through single grid and pMG with restart
- Domain reconstruction selection runs quantized maps and shared or direct quadrature with FP64 evolving fields
- Asynchronous CUDA output freezes the accepted epoch before a new device write

## `dnds_test_cauchy_equation_dofs`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_equation_dofs_np4` |
| 1 | `dnds_cauchy_equation_dofs_np1`, `dnds_cauchy_equation_dofs_np2` |
| 2 | `dnds_cauchy_equation_dofs_np8` |

[test_EquationDofs.cpp](../../test/cpp/Cauchy/test_EquationDofs.cpp)

- Equation schema mapping uses semantic names instead of component count or ordinal
- Whole-equation dispatch preserves named blocks and scalar order across bounded view families
- Fixed equation block views preserve parent pitch and component ranges without changing storage
- Logical flow4/5 with RANS and many species maps exactly across physical layouts and storage groups
- Hybrid equation blocks expose physical grouping and rebind across parent layouts
- Owned-only pointwise algebra and reductions overlap direct halo writes without regional splitting
- Constant-density physical accumulation omits only the algebraic pressure channel
- Semantic validation and writer reservation reject bad bindings without partial publication
- Retained scopes preserve allocation lifetime and storage replacement requires explicit rebinding
- Numeric equation aliases and read scopes retain rows and values without retaining MPI context
- Coupled matrix action ILU SGS and left GMRES preserve separate and permuted logical equation fields
- Logical linear driver rejects inconsistent physical binding before any halo setup
- Logical scalar-diagonal SGS handles all blocks and fails an abandoned partial output
- The same BDF and coupled block solve interface advances all flow RANS and species components
- One time driver advances coupled and separate equation storage with identical coupled stage mathematics
- Queued equation algebra retains temporaries and preserves logical blocks across layouts
- Equation algebra reserves every output before writing and time sources cannot be silently mutated

## `dnds_test_cauchy_equation_source`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_equation_source_np2` |
| 1 | `dnds_cauchy_equation_source_np1` |
| 2 | `dnds_cauchy_equation_source_np4`, `dnds_cauchy_equation_source_np8` |

[test_EquationSource.cpp](../../test/cpp/Cauchy/test_EquationSource.cpp)

- Total mixture inlet preserves prescribed composition stagnation enthalpy and entropy
- Shared transported species repair preserves Euler constants and strided neighbors
- Constant-volume source preserves frozen blocks and matches the SI reactor at nonunit scales
- Equation primitive and mixture boundary producers preserve full energy composition and moving axes
- Mixture thermal references enforce a requested temperature through actual conservative energy
- Attached RANS leaves mixture thermal bounds and chemical Jacobian columns unchanged
- Mixture source point matches the Euler source and all conservative derivative columns
- Mixture physical ALE flux derivatives include composition energy and all conservative columns
- Owned chemical sources honor separate layouts, stale temperature guesses and ghost independence
- Mixture primitive and conservative gradients include composition energy and all kinetic terms
- Mixture viscous transport matches Euler heat and species enthalpy flux with effective coefficients
- Chemical source rejects output reservations and Jacobian aliases before changing values
- Actual conservative chemical sources advance a separate-field implicit reactor toward Cantera

## `dnds_test_cauchy_finite_volume_physics`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_finite_volume_physics_np2` |
| 1 | `dnds_cauchy_finite_volume_physics_np1` |
| 2 | `dnds_cauchy_finite_volume_physics_np4`, `dnds_cauchy_finite_volume_physics_np8` |

[test_FiniteVolumePhysics.cpp](../../test/cpp/Cauchy/test_FiniteVolumePhysics.cpp)

- Equation FV uses the established P3/gradient/WBAP flow5 kernels across CPU launchers and layouts
- Equation FV local stages and physical-field halos inherit the agreed safety policy
- Canonical equation face flux agrees with cell incidence on rotated and self-periodic faces
- Logical nonlinear flow5 retains native finite-sweep stage mathematics
- Equation wall values preserve internal energy and support empty owners on a physical 3D mesh
- Spatial initial input averages conservative polynomials on distributed native layouts
- Uniform inertial ACM flow preserves continuity on a rotating mesh
- Face-centered assembled Jacobians match direct actions in stationary and moving frames
- A single-species material uses empty independent composition and no chemical equation
- Mixture characteristic WBAP consumes fresh distributed cell caches for modal and direct gradients
- Finite-volume admissibility uses one prepared complete mixture policy
- Mixture interface packets retain complete equation rows and reject a non-simplex publication
- Direct P1 mixture PP shares one thermal factor across split flow and species storage
- Distributed prescribed mixture boundaries match independent nonconstant FV fluxes across layouts and launchers
- Mixture isothermal wall conducts the independent Fourier flux without species leakage
- Tagged mixture state initializes distributed Domain and prescribed boundary through one resolver
- Explicit mixture rejects inadmissible raw flow or species and retries without increment compression
- Configured mixture Domains advance every block through production time and publication paths
- Physical mixture Jacobians match frozen-signal residual differences and solve through the logical driver
- Actual mixture FV residual preserves separate/permuted fields and core overlap without chemistry in face queries
- Mixture volume quadrature adds the conservative chemical rate and leaves full energy unchanged
- Shared quadrature keeps chemical source and P3 positivity caches unique without changing results
- Nonconstant assembled mixture diffusion matches an independent face integration with direct backend coefficients
- Nonuniform chemical volume sampling agrees with independent Cartesian Gauss quadrature and direct backend rates
- Equation P3 positivity protects actual EOS samples and preserves unlimited guesses in split and permuted layouts
- Equation workspace agrees PP controls before binding conditional halo fields
- Equation implicit PP compresses RHS or pseudo-time and repairs species without changing flow
- Coupled mixture nonlinear stage recovers a manufactured target through separate or permuted arrays
- Production equation nonlinear stages advance a distributed reacting FV state toward Cantera

## `dnds_test_cauchy_implicit_domain`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_implicit_domain_np2` |
| 1 | `dnds_cauchy_implicit_domain_np1` |
| 2 | `dnds_cauchy_implicit_domain_np4`, `dnds_cauchy_implicit_domain_np8` |

[test_ImplicitDomain.cpp](../../test/cpp/Cauchy/test_ImplicitDomain.cpp)

- Periodic ACM physical step transports velocity and preserves the pressure gauge
- An active ACM pressure outlet suppresses the closed-domain gauge
- ACM BDF observes continuity separately from named momentum convergence
- Production DITR controller uses common limits unless an override is supplied
- Native physical implicit methods preserve uniform periodic flow and transactional commit
- P3 implicit periodic transport conserves flow and discard restores the VR warm guess
- Steady BDF1 infinite-dt mode removes a nonconstant residual without advancing physical time
- Capped steady stages advance by default and reject only with explicit control
- Strict implicit acceptance requires settled VR in addition to a loose mean target
- DITR coupled cap advances by default and rejects only with explicit control
- Rotating physical domains retain a uniform inertial flow with explicit and implicit stage times

## `dnds_test_cauchy_jacobian`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_jacobian_np2` |
| 1 | `dnds_cauchy_jacobian_np1` |
| 2 | `dnds_cauchy_jacobian_np4`, `dnds_cauchy_jacobian_np8` |

[test_Jacobian.cpp](../../test/cpp/Cauchy/test_Jacobian.cpp)

- ACM temporal Jacobian distinguishes pressure pseudo mass from differential physical mass
- ACM Euler scalar wall mode adds only the prepared spectral diagonal
- Scheme A characteristic action reproduces the differentiated mechanical flux
- Rotational periodic ACM matrix differentiates the production face residual in stored axes
- Six-component frozen face leaves retain EOS coupling and additive matrix and strided product semantics
- Integrated approximate Jacobian matches frozen-spectral flux differences on periodic and physical boundaries
- Frozen physical semi-matrix-free blocks agree with assembled action, split SGS and local ILU
- Selected physical Jacobians are principal matrices after periodic frame transformation

## `dnds_test_cauchy_linear`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_sgs_subpartitions_omp_np2` |
| 1 | `dnds_cauchy_linear_np1`, `dnds_cauchy_linear_np2` |
| 2 | `dnds_cauchy_linear_np4`, `dnds_cauchy_linear_np8` |

[test_LinearProblem.cpp](../../test/cpp/Cauchy/test_LinearProblem.cpp)

- Optional inverse refinement repairs the SST action and preserves truncated subspaces
- Native linear action overlaps direct MPI with core and preserves frozen local SGS semantics
- SGS mesh subpartitions freeze cross-block reads in ordinary and zero-start split sweeps
- Linear matrix construction rejects invalid topology and supports empty owners
- Cauchy native ILU copies only local coefficients and validates ordering fill epochs and local parts
- RCM factor ordering preserves local parts and solves the same dense system
- Linear diagonal preparation handles general component counts and reports guarded inverse fallback
- Common block inverse widths preserve a dense solve and guarded fallback
- Staged SGS retains numeric allocations and row metadata after original owners expire
- Queued linear actions retain expired owners and reject a submitted stale inverse

## `dnds_test_cauchy_linear_solver`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_linear_solver_np2` |
| 1 | `dnds_cauchy_linear_solver_abort_np2`, `dnds_cauchy_linear_solver_np1` |
| 2 | `dnds_cauchy_linear_solver_np4`, `dnds_cauchy_linear_solver_np8` |

[test_LinearSolver.cpp](../../test/cpp/Cauchy/test_LinearSolver.cpp)

- SGS and matrix action use direct evaluated products on strided rows
- Jacobi matches independent frozen-remote local-sweep oracle
- Dedicated zero-start SGS needs no old storage and preserves explicit-zero results
- Residual observation cadence preserves stationary iterates and skips only diagnostic reductions
- Residual-stopped SGS publishes zero when a repeated solve needs no sweep
- Cauchy coupled linear solvers match a global dense oracle with real persistent halos
- Logical coupled SGS retains physical-field scan values with remote middle exchange
- GMRES fixed ILU corrections match one Arnoldi step of the distributed dense preconditioner
- Cauchy linear controls and rank-local access failures reject collectively before transfers
- Fixed-count stationary corrections preserve iterates without diagnostic actions or products
- Fixed-count zero-start ILU overwrites dirty solution and handles zero corrections
- Fixed-count final finite check agrees on a rank-local nonfinite update
- Evaluated block publication validates row identity, column ordinals and active read scopes
- CPU AMG preserves distributed action and CPU column-major blocks
- One CPU AMG correction follows logical block order in a permuted equation field

## `dnds_test_cauchy_multidomain`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_multidomain_np4` |
| 2 | `dnds_cauchy_acm_interface_cuda`, `dnds_cauchy_multidomain_np1`, `dnds_cauchy_multidomain_np2`, `dnds_cauchy_multidomain_np8` |

[test_Multidomain.cpp](../../test/cpp/Cauchy/test_Multidomain.cpp)

- Multi-domain BDF2 restart reserves every domain and redistributes by persistent IDs
- Moving ACM disk follows the pressure datum of the connected outer domain through restart
- Moving ACM disk shares one pressure shift across a closed interface
- Coordinated domains retain single-domain time equations and atomic publication
- P3 flow5 crosses a posed disk interface with dense and sparse distributed matching
- P0 and physical-gradient P1 publish complete multidomain traces without modal storage
- DITR freezes distinct rotating midpoint and endpoint interface packets
- Split flow and SA fields cross a rotating interface as one equation packet

## `dnds_test_cauchy_polynomial_multigrid`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_polynomial_multigrid_np2` |
| 2 | `dnds_cauchy_polynomial_multigrid_np1`, `dnds_cauchy_polynomial_multigrid_np4`, `dnds_cauchy_polynomial_multigrid_np8` |

[test_PolynomialMultigrid.cpp](../../test/cpp/Cauchy/test_PolynomialMultigrid.cpp)

- Bottom pMG uses mean convection and jump-only face viscosity in both RHS profiles
- Each coarse smoother inherits defaults and resolves independent partial controls
- Finest FAS acceptance shares selected-component L1 L2 convergence with the stage solver
- Coarse FAS rate reuse and fail-fast halos preserve distributed viscous cycle results
- FAS consumes the measured finest VR rate once for field and logical equations
- Polynomial FAS recovers the original P3 manufactured equation across layouts and distributed halos
- Polynomial FAS fixed points and preflight errors preserve fine means
- Selected pMG recovers nonconstant principal equations while preserving frozen means and full residual reporting
- Cauchy pMG physical discard and reprepare retain the fine reconstruction transaction
- Logical polynomial FAS recovers nonconstant RANS and reacting mixture equations without packing
- Equation Domain performs one spatial FAS update per DITR coupled stage and retries exactly

## `dnds_test_cauchy_rans_fv`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_rans_fv_np2` |
| 1 | `dnds_cauchy_rans_fv_np1` |
| 2 | `dnds_cauchy_rans_fv_np4`, `dnds_cauchy_rans_fv_np8` |

[test_RANSFiniteVolume.cpp](../../test/cpp/Cauchy/test_RANSFiniteVolume.cpp)

- Barth RANS overlay keeps face values fixed when selecting unlimited viscous slopes
- Centered SST source Jacobian is independent of the VR polynomial
- Finite-volume admissibility includes the prepared RANS mean policy
- ACM wall-bounded first implicit correction transports the velocity and turbulence blocks
- Characteristic WBAP transforms only flow5 across combined and segmented equation storage
- Ordered six-component assembled faces match permuted storage and direct actions
- Explicit RANS repairs scalars separately from the uncompressed flow5 update
- RANS wall force uses the completed RHS momentum flux with the physical pressure sign
- RANS FV integrates quadrature sources and their positive implicit destruction diagonal
- Zero-SA wall convection retains the interior SA viscous jump
- SA boundary samples use wall reflection and frame-relative incoming flow
- Two-equation boundary points use mean wall distance and the correct thermal viscosity
- RANS P3 residual and WBAP preserve split layout and core halo scheduling semantics
- Explicit mean gradients exchange current values across layouts and CPU launchers
- High-order SST auxiliary faces retain quadrature-dependent wall transport
- Two-equation auxiliary viscosity consumes current conservative mean gradients
- RANS final thermal increment compression retains Euler whole-vector scaling
- P3 thermal reconstruction compression preserves admissible RANS linear coefficients
- SA reference-quality failure preserves thermally admissible uncompressed coefficients
- RANS reconstruction PP preserves flow and scalar polynomials independently
- SA implicit updates repair scalars after flow compression without unfreezing masked components
- RANS nonlinear stages recover a manufactured viscous source balance
- SA cached corrections retain pre-RHS auxiliary publication across CPU launchers
- P0 jump-only face viscosity retains physical gradients in the SA volume source

## `dnds_test_cauchy_riemann`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_riemann` |

[test_Riemann.cpp](../../test/cpp/Cauchy/test_Riemann.cpp)

- Transverse wave jump uses supplied acoustic data and is side-frame invariant
- Riemann module uses supplied pressure and acoustic data independently of energy
- Energy-origin shifts and oriented reversal preserve the supplied Euler flux convention
- Every flux accepts external thermodynamics and transforms covariantly under energy-origin changes
- Shared Riemann formulas support explicit kernel-local arithmetic types

## `dnds_test_cauchy_riemann_legacy`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_riemann_legacy` |

[test_RiemannLegacy.cpp](../../test/cpp/Cauchy/test_RiemannLegacy.cpp)

- Scalar convection uses the caller's fixed dissipation independently of flow flux
- Every Cauchy Riemann selection matches legacy perfect gas facial and mean state behavior
- Every selected flux exposes explicit common-input requirements and missing data errors
- All Euler fluxes satisfy stationary face consistency and reversal independently of legacy

## `dnds_test_cauchy_state_input`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_state_input` |

[test_StateInput.cpp](../../test/cpp/Cauchy/test_StateInput.cpp)

- ACM velocity and relative pressure use independent physical scales
- Perfect gas molecular laws preserve Euler scaling and all property consumers
- Tagged physical state conversions preserve Euler variants and distinct RANS scales
- Initial boundary and reference adapters share conversion without density fallback
- Mixture state inputs match Euler total and sensible energies with nonunit scales
- Spatial state overrides preserve precedence plane sign and physical expression seeds

## `dnds_test_cauchy_viscous_flux`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_cauchy_viscous_flux_np2` |
| 1 | `dnds_cauchy_viscous_flux_np1` |
| 2 | `dnds_cauchy_viscous_flux_np4`, `dnds_cauchy_viscous_flux_np8` |

[test_ViscousFlux.cpp](../../test/cpp/Cauchy/test_ViscousFlux.cpp)

- P3 viscous face derivatives reproduce polynomial heat conduction and Couette production
- Direct P1 Barth preserves an independently selectable unlimited viscous gradient
- Polynomial viscosity exchanges the selected limited or unlimited coefficients after the last VR commit
- Viscous policy disagreement is rejected before differing transfer schedules are bound
- Unified equation orchestration preserves primitive-gradient flow5 viscosity
- Polynomial wall viscosity has the Euler normal jump and noGRP switch
- Nonzero quadratic temperature does not introduce viscosity at Euler open or slip boundaries

## `dnds_test_device_packing`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_device_packing_np2` |
| 1 | `dnds_device_packing_np1` |

[test_DevicePacking.cu](../../test/cpp/DNDS/test_DevicePacking.cu)

- Device packing preserves wire row order across layouts and persistent transient modes
- Device packing handles CSR zero rows empty source peers and distributed fanout
- Device packed POD uses the registered schema extent and preserves omitted fields and padding
- Device packed collisions require a reducer and use stable receiver reduction
- Device packing begins eagerly and overlaps owned compute with a gated communication stream
- Device packed preflight rejects invalid placement and rolls back unmodified destination

## `dnds_test_distributed_failure`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_distributed_failure_factory_np2`, `dnds_distributed_failure_reducer-completion_np2`, `dnds_distributed_failure_solver-checked-preflight_np2`, `dnds_distributed_failure_solver-unchecked-preflight_np2`, `dnds_distributed_failure_solver-worker_np2`, `dnds_distributed_failure_transfer_np2` |
| 2 | `dnds_distributed_failure_factory_np4`, `dnds_distributed_failure_factory_np8`, `dnds_distributed_failure_reducer-completion_np4`, `dnds_distributed_failure_reducer-completion_np8`, `dnds_distributed_failure_solver-checked-preflight_np4`, `dnds_distributed_failure_solver-checked-preflight_np8`, `dnds_distributed_failure_solver-unchecked-preflight_np4`, `dnds_distributed_failure_solver-unchecked-preflight_np8`, `dnds_distributed_failure_solver-worker_np4`, `dnds_distributed_failure_solver-worker_np8`, `dnds_distributed_failure_transfer_np4`, `dnds_distributed_failure_transfer_np8` |

[test_DistributedFailure.cpp](../../test/cpp/Cauchy/test_DistributedFailure.cpp)


## `dnds_test_id_row_transfer`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_id_row_transfer_np2` |
| 1 | `dnds_id_row_transfer_np1` |
| 2 | `dnds_id_row_transfer_np4`, `dnds_id_row_transfer_np8` |

[test_IdRowTransfer.cpp](../../test/cpp/DNDS/test_IdRowTransfer.cpp)

- Exact persistent-ID lookup transfers native rows after arbitrary rank distribution and local permutation
- ID row transfer rejects duplicate sources missing targets and nonbijective coverage

## `dnds_test_native_dof_algebra`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_native_dof_algebra_np2` |
| 1 | `dnds_native_dof_algebra_np1` |
| 2 | `dnds_native_dof_algebra_np4`, `dnds_native_dof_algebra_np8` |

[test_NativeDofAlgebra.cpp](../../test/cpp/DNDS/test_NativeDofAlgebra.cpp)

- Operator input layout caches track queued mutations and storage aliases
- Independent OMP row kernels return deterministic first numerical failure
- Reserved DoF scratch reuses prepared level aliases and communicating slots
- Native DOF algebra operates on logical owned entries for arbitrary table widths
- Native DOF fixed widths share handles and numerical copies are explicit
- Native DOF algebra validates semantic spaces, immutable shape and overlap
- Fixed DOF component contract survives dynamic Array imports and replacement
- Immediate DOF convenience rejects pending or active access before using an inline lane
- Native DOF commands retain values and honor explicit cross-stream joins
- Native DOF global reductions include uneven and empty owned ranks with collective failure
- Native DOF operators support a manufactured diagonal Krylov solve
- Native DOF commands and reductions mix physical formats without touching ghosts or padding
- DOF halo publication tracks completed epochs while owned core work overlaps

## `dnds_test_nccl_transfer`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_nccl_transfer_np2` |
| 1 | `dnds_nccl_transfer_np1` |

[test_NCCLTransfer.cu](../../test/cpp/DNDS/test_NCCLTransfer.cu)

- NCCL packed rows match exact values across layouts and queued producer consumer epochs
- NCCL direct contiguous copies and explicit unsupported bindings
- NCCL packed CSR empty ownership and omitted POD bytes preserve receiver contents
- NCCL enqueue overlaps core work with gated transfer and retains leases and workspace
- CUDA producer chains device-packed MPI and NCCL across streams through pending shutdown
- NCCL global Array reduction orders producer and consumer across streams
- NCCL DOF halo publishes the retired producer version without a submission host wait
- Unchecked NCCL missing producer edge demonstrably transfers a stale epoch
- NCCL collisions and aborted communicator reject explicitly
- NCCL binding checks communicator identity beyond congruent MPI groups

## `dnds_test_runtime_array`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_array` |

[test_RuntimeArray.cpp](../../test/cpp/DNDS/test_RuntimeArray.cpp)

- Native row teams use the same explicit policy on controller and CPU worker
- Native setup failures are emitted after all row workers join
- Reader ingress and distributed row teams retain independent policies
- Array mutation epochs cover aliases and canceled reservations without claiming completion
- Allocation diagnostics count shared capacity once and do not retain arrays
- Whole-array residency survives projections without changing row access
- Scheduled operands publish placement in borrowed whole-array views
- Kernel spans preserve signed slices and metadata publications own their storage
- Unused Array access cancellation preserves values validity and generation
- Runtime row layouts reject unsigned shapes that cannot be indexed by signed views
- Generated immutable metadata preserves ordinal values and abandons failed construction
- Worker eager submissions retain all five typed policies and generic launcher compatibility
- Five layout policies produce distinct typed views and share dynamic storage
- Uniform variable layouts project to typed tables in direct and eager access
- Static row policy reaches typed access and preserves dynamic interoperability
- Static width and capacity policies validate empty layouts and physical stride
- Static maximum capacity retains variable logical widths and untouched padding
- Array allocation accounting retains capacity and deduplicates aliased metadata
- Consumed Array handles reject queries while retained access survives move and release
- Native failed write scope invalidates its aliases
- native Array retains row layout and allocation across reset
- native ArrayPair core only needs owned access
- Fully fixed row spans use one pointer and general rows preserve borrowed access
- Native sized row views preserve logical extents and retained metadata
- Native fixed POD rows share read and write span access
- native layout zero rows padding and overflow validation
- native typed handles share one allocation and logical metadata record
- Imported six-equation state retains external owner and one access domain
- Buffer address queries detect foreign aliases without acquiring access
- Imported read-only storage rejects writes without invalidating readers
- Independent reconstruction fields share buffered CSR metadata and retain old shape
- Known-row construction fills independently and reports worker failures
- Physical tables preserve logical rows across padding and tile boundaries
- Physical projections are checked once and retain specialized component strides
- Kernel projections preserve fixed row extents and independent padded owned ghost pitches
- Pair views keep independent column pitches and eager workers retain physical layout
- Allocation alignment is provider-enforced and over-aligned POD survives placement snapshots
- Local Eigen matrices honor physical component stride without packing
- Explicit logical conversion preserves values and retained source views
- Checked element projections reject invalid row and component before memory access
- Two disjoint row ranges retain supplied traversal order and offsets

## `dnds_test_runtime_cuda_engine`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_cuda_engine` |

[test_CUDAExecution.cu](../../test/cpp/DNDS/test_CUDAExecution.cu)

- CUDA eager operands retain whole-array placement through kernel submission
- Retained CUDA contexts and immediate leases survive a drained engine close
- CUDA mutation epochs precede completion and survive stream retirement
- CUDA self waits reuse the queued prefix and retain operands until completion
- Closing mixed CPU CUDA execution retains temporary operands and observes completed events
- CUDA dynamic multilevel temporaries join independent readers before final stream release
- CUDA mock communication and core kernels are simultaneously resident
- CUDA missing eager join returns stale values while ordered consumption is correct
- Checked CUDA eagerly queues dependent kernels and rejects missing prefixes and free joins
- Eager CUDA uses explicit prefix events with retained DMA owners and a three stream join
- CUDA owned CSR stencil finishes while ghost writes are pending then joins ring work
- CUDA fixed prefixes retire prior batches and preserve delayed ownership
- CUDA launch exceptions drain submitted uses and invalidate partial and skipped writes
- Concurrent CUDA host submitters preserve one native stream order without kernel events
- CUDA dependency only prefixes retire earlier batches and completed history is discarded
- CUDA async allocation supports dynamic tables and prepared padded CSR metadata behind pending work
- Backend allocation resources receive the final selected free stream across independent handles
- CUDA aligned physical tables cross streams and release after the final use

## `dnds_test_runtime_element_schema`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_element_schema` |

[test_RuntimeElementSchema.cpp](../../test/cpp/DNDS/test_RuntimeElementSchema.cpp)

- ElementSchema registration is MPI-free and shares canonical native definitions
- ElementSchema swaps each mixed scalar word and preserves all padding
- ElementSchema rejects invalid field layouts and incompatible registrations
- ElementSchema retains vended MPI structs after caller handles are freed
- ElementSchema marks unflattened MPI typemaps native opaque
- Registered ElementSchema supports typed transfer and custom reduction

## `dnds_test_runtime_engine`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_engine` |

[test_EagerEngine.cpp](../../test/cpp/DNDS/test_EagerEngine.cpp)

- CPU stream resources survive close and immediate fences respect worker execution
- Failed CPU retirement cannot publish a successful late write lease
- Array submission captures survive launch callback disposal until retirement
- Controller resources survive inline or worker work and retire on the controller
- Completion notifications wake CPU lanes without controller polling
- Immediate context borrows prevent closure and weak runtime handles reject expiry
- Context shutdown progresses controller transport and CPU lanes before releasing owners
- Drain waits autonomous prefixes by priority only after controller progress retires
- Active mock communication overlaps inline and worker core kernels
- Missing eager join produces a deterministic stale result without a data race
- Manufactured triangular sweeps preserve in-place ordering and eager stopping
- Inline and worker launchers execute the same eager fork join loop
- Record describes a fixed prefix and a waiting lane cannot exhaust workers
- Imported CPU prefixes progress without blocking another executor
- Queued temporary ownership and releases follow CPU completion on either lane
- Errors skip dependent work but retain ordered cleanup and drain other lanes
- Concurrent host submitters linearize commands and preserve every accepted use
- Failed lanes still honor joins before releasing shared resources
- Rejected release returns the caller handle and never guesses a stream
- Array reservations publish at execution and preserve same stream read write ordering
- Owned only CSR stencil can complete while ghost receives remain pending
- Pending operands retain old values and buffered row sizes across handle reset
- Dynamic array batches retain segmented fields and order queued producers
- Dynamic array batch failure invalidates every declared output
- Rejected reservations cancel while failed or skipped writes stay invalid
- Eager byte copies use the same pending access contract on inline and worker CPU
- CPU async allocation facade preserves shared variable shape metadata and handle ownership

## `dnds_test_runtime_native_io`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_native_io_np2` |
| 1 | `dnds_runtime_native_io_np1` |
| 2 | `dnds_runtime_native_io_np4`, `dnds_runtime_native_io_np8` |

[test_RuntimeNativeIO.cpp](../../test/cpp/DNDS/test_RuntimeNativeIO.cpp)

- Native payload uses caller buffers across dense CSR padded fixed and static-max shapes
- JSON byte segments support in-memory DOM and retain file mode checks
- Opposite-endian mixed records swap individual declared words and preserve object padding
- Malformed native records reject before payload fill and permit pristine retry
- Malformed H5 native metadata rejects collectively without truncated CSR association
- Native collective checkpoint rejects one rank's moved or unpublished Array before file mutation
- Version two legacy byte CSR remains readable through direct native allocation

## `dnds_test_runtime_row_builder`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_row_builder` |

[test_RuntimeRowBuilder.cpp](../../test/cpp/DNDS/test_RuntimeRowBuilder.cpp)

- Row builder rejects overflowing requests before changing existing rows
- Known shape rows fill one final table or CSR and reject failed publication
- Exclusive padded edits block sharing and retain value addresses across Commit
- Exclusive edits reject aliases pins and independently shared metadata
- Abandoned exclusive edits fail closed and table widths survive physical padding
- Parallel incremental CSR construction packs once at publication
- Padded row builder updates logical sizes within fixed capacity
- Detached shape drafts preserve old aliases and abandon without publication
- Builder empty shapes and POD values keep element alignment

## `dnds_test_runtime_row_transfer`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_row_transfer_np4` |
| 1 | `dnds_runtime_preflight_off_failure_np2`, `dnds_runtime_row_transfer_abort_np2`, `dnds_runtime_row_transfer_np1`, `dnds_runtime_row_transfer_np2` |
| 2 | `dnds_runtime_row_transfer_np8` |

[test_RuntimeRowTransfer.cpp](../../test/cpp/DNDS/test_RuntimeRowTransfer.cpp)

- Prepared CPU transfer groups retain separate payloads and perform one optional agreement
- Grouped reservation rejection releases all fields before any payload is posted
- Optional collective preflight removes only successful launch reductions
- Managed MPI requests survive dropped handles and progress worker consumers during close
- Inherited named channels share a communicator and preserve concurrent payloads
- Communication context adoption and policy aliases retain one resource and tag namespace
- Communication context agrees its immutable safety policy before preparing routes
- Fail-fast MPI launches preserve padded rows in eager and checked execution
- Large permuted routes preserve row order padding and rank-local collision flags
- Complete small row relations agree with independent sorted collision oracle
- Typed binding rejects same-size arithmetic schema confusion
- Sender ordering derives a checked route without changing peer row correspondence
- Native direct ragged halo retains leases and reusable routing
- Inferred payload shapes retain sparse holes widths and collective policy checks
- Table width agreement preserves sparse rows and handles rank-varying layouts
- Default inferred transfers preserve table stride padded capacity and CSR through repeated halos
- Preserved inference retains typed empty ranks and rejects remote rows that do not fit
- Preserved sparse table slots keep width and explicit compact retains zero-length holes
- Binding rejects reordered communicator and rank-local tag reservation failure
- Combined payload and binding preflight reject rank-local mismatches before requests
- Typed MPI bindings reject one unpublished shape collectively and recover after commit
- Direct permutation honors imported buffer origins and padded state strides
- One route transfers padded reconstruction rows into a packed CSR destination
- Native reverse fan-in rejects copy and supports reductions
- Native POD transfers use MPI extent and support custom reductions
- Native empty routes and generation validation
- Native CUDA reduction dispatch covers wide rows without requiring device MPI
- Native CUDA integer reduction families agree with the CPU reference
- Native CUDA direct MPI and device fan-in reduction
- Mixed physical tables use direct datatype routes and preserve logical rows
- Strided receive collisions use scalar contribution groups for builtin and custom POD reductions
- Communication worker preserves repeated packed and direct sparse transfers
- Communication worker overlaps core reads with active receiver completion
- DevicePacked is explicit and rejects host endpoints collectively
- MPI enqueue and unavailable NCCL capabilities reject explicitly
- Empty direct routes retain stream ordering without payload access or host readiness
- Locally empty direct participants retain the nonempty peer preflight sequence

## `dnds_test_runtime_transient`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_runtime_transient_np2` |
| 1 | `dnds_runtime_transient_np1` |
| 2 | `dnds_runtime_transient_np4`, `dnds_runtime_transient_np8` |

[test_RuntimeTransient.cpp](../../test/cpp/DNDS/test_RuntimeTransient.cpp)

- Known route rejects negative extents on every rank before publishing
- MPI allreduce queues behind a producer and orders the next reduction and consumer
- Foreign buffer aliases cannot bypass copy or collective reservations
- Direct transfers support SELF and reordered split communicators without inherited tag attributes
- Transient reduction rejects missing collision resolution before publishing
- Transfer completion imported into eager progress orders its consumer
- CUDA eager producer portable packed MPI and consumer retain ordering
- Transient and persistent contexts transfer permuted SoA rows
- Dynamic CSR migration privately prepares shapes and publishes once
- Dynamic migration preserves a strided table instead of publishing temporary CSR
- Dynamic migration rejects a missing source collectively and leaves the route usable
- Main-thread MPI progression joins worker computation with scheduled leases
- Checked MPI rejects a rank-local missing reader edge collectively and permits retry
- MPI completes while a core worker is active across Euler field widths
- Rank-local payload and lease mistakes fail collectively without changing values
- Repeated halo gathers and reverse residual assembly share routes across fields
- Uneven empty-rank field bundles migrate out and back with epoch invalidation

## `dnds_test_sparse_batch_exchange`

| Level | CTest registrations |
|---|---|
| 0 | `dnds_sparse_batch_exchange_np4` |
| 1 | `dnds_sparse_batch_exchange_np1`, `dnds_sparse_batch_exchange_np2` |
| 2 | `dnds_sparse_batch_exchange_np8` |

[test_SparseBatchExchange.cpp](../../test/cpp/DNDS/test_SparseBatchExchange.cpp)

- Sparse fixed-record batches fan out and return variable proposal counts
- Empty batch peers complete and invalid record layouts fail collectively

## `euler_test_evaluator_pipeline`

| Level | CTest registrations |
|---|---|
| 0 | `euler_evaluator_small_np2` |
| 2 | `euler_evaluator_pipeline_np1`, `euler_evaluator_pipeline_np2`, `euler_evaluator_pipeline_np4`, `euler_evaluator_pipeline_np8` |

[test_EulerEvaluator.cpp](../../test/cpp/Euler/test_EulerEvaluator.cpp)

- Existing Euler configurations retain full mesh audits when new keys are absent
- EulerEvaluator pipeline: IV (NS, P1, 2D)
- EulerEvaluator pipeline: NACA0012 (NS_SA, P1)
- EulerEvaluator pipeline: SA IDDES with runtime ft2 disabled
- EulerEvaluator pipeline: Box (NS_3D, P1)
- RANS wall cache rejects changed geometry and metric publications
- Euler two-equation turbulent normal stress follows the physical face normal
- Viscous wall perturbations match prescribed turbulence ghost values
- SA GG boundaries preserve means, face targets and traversal independence
- EulerP scalar GG reads neighbour cells and face-indexed boundary buffers
- Euler reconstruction PP keeps RANS out of flow beta and order fallback
- Euler gradients: EvaluateDt prepares limited buffers and direct FV explicitly reuses them
- LVR target and branch halos agree across 2-D MPI partitions
- LVR explicit gradient reuse matches a fresh Barth or WBAP reference
- EulerSolver static reconstruction mask owns selectors and setup storage
- EulerSolver static reconstruction mask builds a 3D coordinate selection
- EulerSolver static reconstruction mask survives limiter paths and time updates
- Euler distributed volume output reads native authority after detached facade metadata edits
- Native Euler ingress restart and serial boundary output retain field association
- Native Euler H5 restarts preserve sparse high Cell IDs and selected variable dimensions
- Euler archived mesh setup refreshes loaded wall vectors after actual coordinate snapping
- Euler periodic source setup accepts configured near-plane cleanup after rotation

## `euler_test_gas_thermo`

| Level | CTest registrations |
|---|---|
| 0 | `euler_gas_thermo` |

[test_GasThermo.cpp](../../test/cpp/Euler/test_GasThermo.cpp)

- IdealGasThermal: standard quiescent air
- IdealGasThermal: Mach 2 flow
- IdealGasThermal: gammaEq controls pressure and gamma controls acoustic speed
- Shared enthalpy helper does not double-count base energy
- Cons2Prim and Prim2Cons round-trip: 3D
- Cons2Prim and Prim2Cons round-trip: 2D
- Cons2Prim and Prim2Cons round-trip preserves base-energy offset
- Prim2Cons: known state verification
- EulerGas eigenvectors: L*R = I for 3D
- EulerGas eigenvectors: L*R = I for non-trivial velocity
- IdealGas convenience eigenvector wrappers produce L*R=I
- EulerGas left eigenvector inverse uses H/a consistency for split gamma
- EulerGas eigenvectors: L*R = I for 2D split gamma
- GasInviscidFlux: x-direction, quiescent gas
- GasInviscidFlux: x-direction, moving gas
- GasInviscidFlux_XY: n=(1,0,0) equals GasInviscidFlux
- IdealGasUIncrement: finite-difference verification
- IdealGasUIncrement: base-energy increment reduces pressure increment
- GetRoeAverage: identical states give same state
- GetRoeAverage: split gamma keeps pressure closure separate from acoustic gamma
- GetRoeAverage: density is geometric mean
- RoeFluxIncFDiff: entropy wave strength preserves split-gamma identity
- GradientCons2Prim: finite-difference verification
- GradientCons2Prim: species base-energy gradient corrects pressure gradient
- CompressionRatio: zero increment gives alpha=0 (no compression needed)
- CompressionRatio: alpha in [0,1]
- ViscousFlux: zero gradient produces zero flux

## `euler_test_physics_properties`

| Level | CTest registrations |
|---|---|
| 0 | `euler_physics_properties` |

[test_PhysicsProperties.cpp](../../test/cpp/Euler/test_PhysicsProperties.cpp)

- PhysicsProperties non-reactive state conversions
- PhysicsProperties non-reactive conservativeThermal returns closure and acoustic gamma
- PhysicsProperties non-reactive total-to-static conversion is closed form
- PhysicsProperties non-reactive static-to-total conversion is closed form
- PhysicsProperties non-reactive cons code-phys round-trip with non-default scales
- PhysicsProperties non-reactive prim code-phys round-trip with RANS and non-default scales
- PhysicsProperties reactive state conversions
- PhysicsProperties reactive conservativeThermal separates gammaEq from acoustic gamma
- PhysicsProperties reactive transport uses Cantera mixture coefficients in code units
- PhysicsProperties reactive temperature gradient matches finite differences
- PhysicsProperties reactive total-to-static conversion iterates mixture thermodynamics
- PhysicsProperties reactive static-to-total inverts total-to-static
- PhysicsProperties reactive total-to-static default options converge or throw
- PhysicsProperties cons code-phys round-trip multi-species with non-default scales
- PhysicsProperties prim code-phys round-trip multi-species with RANS and non-default scales
- PhysicsProperties RANS prim scale factors for all models
- PhysicsProperties RANS cons scale factors
- PhysicsProperties scaleRansPrim round-trip
- PhysicsProperties scaleRansCons round-trip
- PhysicsProperties StateValueOrigin round-trip single-species
- PhysicsProperties StateValueOrigin round-trip multi-species Cantera
- PhysicsProperties cons code-phys linearity and range
- PhysicsProperties prim code-phys linearity and range
- PhysicsProperties RANS scale linearity with multiple scales
- PhysicsProperties default-RANS nRANSVars=0 cons and prim helpers
- PhysicsProperties reference derived scales
- StateValue JSON round-trip

## `euler_test_rans`

| Level | CTest registrations |
|---|---|
| 0 | `euler_rans` |

[test_RANS.cpp](../../test/cpp/Euler/test_RANS.cpp)

- GetMut_KOWilcox: golden value
- GetMut_SST: golden value
- GetMut_RealizableKe: golden value
- GetSource_KOWilcox: zero gradient finite and has destruction
- GetSource_SST: zero gradient finite
- GetSource_RealizableKe: zero gradient finite
- GetSource_KOWilcox: shear gradient golden
- GetSource_SST: shear gradient golden
- GetSource_SST: omega production blends inner and outer coefficients
- SST-2003 uses strain in transport and one capped production in both equations
- GetSource_RealizableKe: shear gradient golden
- GetSource_KOWilcox mode=1: implicit diagonal non-negative
- GetSource_SST mode=1: implicit diagonal non-negative
- SST optional sink diagonal matches frozen-gradient derivatives and preserves RHS
- GetVisFlux_KOWilcox: zero gradient -> zero flux
- GetVisFlux_SST: zero gradient -> zero flux
- GetVisFlux_RealizableKe: zero gradient -> zero flux
- GetVisFlux_KOWilcox: k-gradient produces k-flux
- GetMut_KOWilcox: mut increases with k
- GetMut_SST: mut increases with k
- GetMut_KOWilcox: very small k/omega produces finite mut
- GetMut_SST: very small k/omega produces finite mut
- GetMut_RealizableKe: very small k/eps produces finite mut
- RANS configs: defaults and JSON round-trip preserve historical limits
- Euler evaluator settings: RANS configs are flat nested sections
- GetSource_SA: production limit changes the physical source
- GetSource_SA: SA-R suppresses rotation rather than strain in 2D and 3D
- SA DES lengths: coefficient placement, shielding and disabled scale
- GetSource_SA: DES modes and ft2 switch agree with analytic sources in 2D/3D
- shared SA: explicit frame vorticity matches corrected velocity gradients

## `euler_test_riemann_solvers`

| Level | CTest registrations |
|---|---|
| 0 | `euler_riemann_solvers` |

[test_RiemannSolvers.cpp](../../test/cpp/Euler/test_RiemannSolvers.cpp)

- AUSM family: consistency, face reversal, ALE and untouched passive entries
- AUSM family: contacts and supersonic upwind flux
- AUSM family: exact stationary Mach-20 normal shock
- AUSMPW+ zero-velocity pressure jump matches independent closed form
- AUSM family: split-gamma and base-energy consistency
- AUSM dispatcher: facial flux and mean-state wave estimates are separate
- AUSMPW+ has no growing linear acoustic mode near the Sod plateau
- Roe consistency: identical states give exact flux
- HLLC consistency: identical states give exact flux
- HLLEP consistency: identical states give exact flux
- Variable-gamma Riemann solvers use gammaEq pressure and gamma acoustic paths
- Variable-gamma Riemann solvers remain symmetric with base energy
- Roe base-energy mode reconstructs identity when all lambdas are one
- Roe base-energy contact has no spurious momentum flux
- Roe variants M1-M9 consistency
- Roe M8 and M9 produce finite Sod fluxes
- Roe M8 and M9 batch dispatch matches scalar dispatch
- Roe symmetry: F(UL,UR,n) = -F(UR,UL,-n)
- HLLC symmetry: F(UL,UR,n) = -F(UR,UL,-n)
- Sod shock tube: flux is finite and bounded
- Golden flux values for mixed-state test vector
- All solvers: quiescent gas produces same flux (p-only)
- Roe eigenvalue output: lam0 < lam123 < lam4 for subsonic
- Roe consistency: diagonal normal
- EulerP Roe entropy threshold uses total speed and preserves orientation and rotation

## `euler_test_rotating_frame`

| Level | CTest registrations |
|---|---|
| 0 | `euler_rotating_frame` |

[test_RotatingFrame.cpp](../../test/cpp/Euler/test_RotatingFrame.cpp)

- Extended Euler rotation uses absolute momentum and its negative-source Jacobian

## `euler_test_source_chemical`

| Level | CTest registrations |
|---|---|
| 0 | `euler_source_chemical` |

[test_ChemicalSource.cpp](../../test/cpp/Physics/test_ChemicalSource.cpp)

- ChemicalSource::productionRates — RHS signs at active T
- ChemicalSource::productionRatesAndJacobian — Jacobian sign convention
- ChemicalSource::maxChemicalStiffnessRate — matches fixed-density mass-fraction diagonal
- Constant-volume chemical response agrees with directional differences and source scaling
- ChemicalSource::mixtureR — gas constant correctness

## `geom_test_distributed_boundary_interface`

| Level | CTest registrations |
|---|---|
| 0 | `geom_distributed_boundary_interface_np2` |
| 1 | `geom_distributed_boundary_interface_np1` |
| 2 | `geom_distributed_boundary_interface_np4`, `geom_distributed_boundary_interface_np8` |

[test_DistributedBoundaryInterface.cpp](../../test/cpp/Geom/test_DistributedBoundaryInterface.cpp)

- Distributed sliding matches exact Cartesian donors across geometry and cell owners
- Coarse boxes retain reference-tolerance projections outside physical endpoints
- Distributed matching rejects overlapping interiors and inconsistent settings collectively

## `geom_test_elements`

| Level | CTest registrations |
|---|---|
| 0 | `geom_elements` |

[test_Elements.cpp](../../test/cpp/Geom/test_Elements.cpp)

- Audit regression: periodic row rvalue assignment copies values
- Audit regression: OpenFOAM comments preserve the following token
- ElementTraits: standard coordinates have correct dimensions
- ElementTraits: face definitions are consistent for 2D elements
- ElementTraits: face definitions are consistent for 3D elements
- ElementTraits: ExtractFaceNodes works correctly
- ElementTraits: order elevation data is consistent for O1 elements
- ElementTraits: order elevation data for specific elements
- ElementTraits: O2 elements do not have further elevation
- ElementTraits: ExtractElevNodeSpanNodes works correctly
- ElementTraits: bisection data is valid for O2 elements
- CGNS PENTA_18 connectivity preserves the reference prism mapping
- VTK connectivity selects distinct input nodes and preserves the external conventions
- Shape functions: partition of unity
- Shape functions: nodal interpolation (Kronecker delta)
- Shape functions: D1 derivatives vs finite difference
- Shape functions: D2 derivatives vs finite difference of D1
- Shape functions: all derivatives are finite
- Face normals: 2D elements match expected unit normal at quad points
- Face normals: 3D elements match expected unit normal at quad points
- Face normals: CGNS outward convention (no centroid correction needed)
- Edge topology: 3D elements have correct edge count and types
- Edge topology: extracted edge nodes are valid cell nodes
- Edge topology: edges are unique (no duplicate edges per cell)

## `geom_test_mesh_cgns_multizone`

| Level | CTest registrations |
|---|---|
| 0 | `geom_mesh_cgns_multizone` |

[test_MeshCGNSMultiZone.cpp](../../test/cpp/Geom/test_MeshCGNSMultiZone.cpp)

- CGNS multizone one-sided connectivity deduplicates 2x2 and 3x3 blocks
- Audit regression: CGNS single-entry boundary PointList

## `geom_test_mesh_connectivity_ghost`

| Level | CTest registrations |
|---|---|
| 0 | `geom_mesh_connectivity_ghost_np2` |
| 1 | `geom_mesh_connectivity_ghost_np1` |
| 2 | `geom_mesh_connectivity_ghost_np4`, `geom_mesh_connectivity_ghost_np8` |

[test_MeshConnectivity_Ghost.cpp](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp)

- Compiled ghost programs expose immutable inspection and own an exact execution signature
- Ghost program agreement rejects different endpoint markers before traversal
- evaluateGhostTree: single hop cell2cell
- evaluateGhostTree: two-hop cell2cell2node with manual ghost
- evaluateGhostTree: 2-ring cell chain
- evaluateGhostTree: union of multiple chains
- evaluateGhostTree: np=1 produces no ghosts
- evaluateGhostTree: performance benchmark
- evaluateGhostTree: doubly-periodic tiled grid
- evaluateGhostTree: small periodic tiled grid
- GhostChain: defaultPrimary compiles
- GhostChain: prefix merging
- GhostChain: invalid chain detection
- GhostChain: dump
- GhostChain: face ghost chain
- GhostChain: AdjacencyDescriptor equality and hash
- GhostChain: entityDepth
- GhostChain: adjKindName
- GhostChain: checkAvailable native registry rejects omitted required relation

## `geom_test_mesh_locality_ordering`

| Level | CTest registrations |
|---|---|
| 0 | `geom_mesh_locality_ordering_np2` |
| 1 | `geom_mesh_locality_ordering_np1` |
| 2 | `geom_mesh_locality_ordering_np4`, `geom_mesh_locality_ordering_np8` |

[test_MeshLocalityOrdering.cpp](../../test/cpp/Geom/test_MeshLocalityOrdering.cpp)

- First-touch node and face permutations preserve distributed cell geometry
- Subregion cell ordering splits conservative core before grouped halo rows
- Locality plans preserve first-touch order from cell or node anchors

## `geom_test_mesh_preflight`

| Level | CTest registrations |
|---|---|
| 0 | `geom_mesh_preflight_np2` |
| 1 | `geom_mesh_preflight_np1` |
| 2 | `geom_mesh_preflight_np4`, `geom_mesh_preflight_np8` |

[test_MeshPreflight.cpp](../../test/cpp/Geom/test_MeshPreflight.cpp)

- Cold mesh preparation shares an explicit policy without changing connectivity or halos
- CGNS owned import partition refinement reorder and publication inherit preflight
- Periodic owned topology and publication preserve disabled preflight
- Conflicting selections of agreed mesh contexts reject before optional preparation stages

## `geom_test_native_connectivity`

| Level | CTest registrations |
|---|---|
| 0 | `geom_native_connectivity_np2` |
| 1 | `geom_native_connectivity_np1` |
| 2 | `geom_native_connectivity_np4`, `geom_native_connectivity_np8` |

[test_NativeConnectivity.cpp](../../test/cpp/Geom/test_NativeConnectivity.cpp)

- Native interpolation retains ordered cones and periodic self incidences with empty ranks
- Native GhostSpec discovers multilayer remote rows and publishes shared routes
- Branched ghost tree keeps frontiers separate and unions endpoint kinds
- Collected intermediate owned nodes remain available to downstream ghost branches
- Native ghost publication agrees complete same-level relation schedules
- Exact payload aliases preserve named descriptors with rank-local alias differences
- Aliased adjacency reads preserve constraints order and independent lease lifetimes
- Global adjacency validation agrees with an independent scalar oracle
- Adjacency self checks retain ghost-only source numbering on empty owners
- Native 3D edge interpolation retains four-parent incidence and empty owners
- Current authority feeds native setup without aliasing detached native setup copies
- Native periodic distributed interpolation preserves owner frames and ordered companions
- Traversal scratch preserves independent publications and exact rows after tree growth
- Entity ghost mapping rejects rank-local unpublished shapes and remains reusable
- Named native relations distinguish incidence any all exact and face versus node neighbors
- Native interpolation matches repeated canonical IDs after XOR permutes frame order
- Independent setup expansions preserve row order and join failures

## `geom_test_native_mesh_periodic_import`

| Level | CTest registrations |
|---|---|
| 0 | `geom_native_mesh_periodic_import_np2` |
| 1 | `geom_native_mesh_periodic_import_np1` |
| 2 | `geom_native_mesh_periodic_import_np4`, `geom_native_mesh_periodic_import_np8` |

[test_NativeMeshPeriodicImport.cpp](../../test/cpp/Geom/test_NativeMeshPeriodicImport.cpp)

- Native serial periodic canonicalization preserves high-order image geometry with compact canonical IDs
- Periodic ingress preserves an explicit ID reservation without reserving discarded source labels
- Native serial periodic identification rejects incomplete ambiguous and fixed-point pairings without changing source
- Native periodic canonicalization uses rigid rotations and labels all original node occurrences
- Native periodic CGNS import builds directly on a nonzero reader and retains frames through transition and archive

## `geom_test_native_mesh_transition`

| Level | CTest registrations |
|---|---|
| 0 | `geom_native_mesh_transition_np2` |
| 1 | `geom_native_mesh_transition_np1` |
| 2 | `geom_native_mesh_transition_np4`, `geom_native_mesh_transition_np8` |

[test_NativeMeshTransition.cpp](../../test/cpp/Geom/test_NativeMeshTransition.cpp)

- Owned native capture assembles after source release with retained companions and exact descriptor rows
- Native assembler rejects rank-local ID schema and closure schedule errors before publication
- Owned native capture collectively rejects mismatched source inventory before ghost-map collectives
- Native transition preserves exact persistent topology and coordinate stencil across independent placements
- Native transition follows complete remote node stars and retains unchanged family order semantics
- Native transition routes empty owners and rank-skewed payload without global staging
- Native prepared transition stages field histories atomically and binds entity fields only on publication
- Native transition distinguishes exact aliases from rank-local candidate shape state and order differences
- Native aliased transition retains repeated periodic IDs and ordered frames
- Owned primary reorder retains IDs and cones when other ranks own no cells
- Canonical and named adjacency storage survives capture assembly migration and local conversion
- Sender partition ordering preserves strided companions through gather and reverse scatter

## `geom_test_native_mesh_wall_dist`

| Level | CTest registrations |
|---|---|
| 0 | `geom_native_mesh_wall_dist_np2` |
| 1 | `geom_native_mesh_wall_dist_np1` |
| 2 | `geom_native_mesh_wall_dist_np4`, `geom_native_mesh_wall_dist_np8` |

[test_NativeMeshWallDist.cpp](../../test/cpp/Geom/test_NativeMeshWallDist.cpp)

- Disabled-preflight wall distance keeps the analytic remote-wall result
- Native wall distance uses remote boundary selection and retains geometry identity
- Wall distance rejects a rank without native authority collectively
- Wall vectors rebuilt after near-plane cleanup use the final shape and configured selector
- Native curved wall preserves method conventions and remote midpoint zeros
- Wall distance rejects an unpublished mesh collectively

## `geom_test_quadrature`

| Level | CTest registrations |
|---|---|
| 0 | `geom_quadrature` |

[test_LocalPerformance.cpp](../../test/cpp/Geom/test_LocalPerformance.cpp)

- Local performance: quadrature cache has one owner and exact values
- Local performance: immutable boundary lookup preserves mutable defaults

[test_Quadrature.cpp](../../test/cpp/Geom/test_Quadrature.cpp)

- Public cached shape access matches direct element evaluation
- Quadrature: scheme constants are valid
- Quadrature: weight sums equal reference element volumes
- Quadrature: Line integrates polynomials up to expected order
- Quadrature: Quad integrates polynomials up to expected order
- Quadrature: Hex integrates polynomials up to expected order
- Quadrature: Triangle integrates polynomials up to expected order
- Quadrature: Tetrahedron integrates polynomials up to expected order
- Quadrature: Prism integrates polynomials up to expected order
- Quadrature: Pyramid integrates constant and low-degree polynomials
- Quadrature: Quadrature class Integration method works
- Quadrature: GetQuadraturePointInfo returns correct data
- Quadrature: IntegrationSimple overloads work correctly
- Quadrature: Taylor basis integration on 2D elements (degree <= 3)
- Quadrature: Taylor basis integration on 3D elements (degree <= 3)
- Quadrature: Taylor basis integration on 1D elements (degree <= 3)
- Quadrature: Pyramid Taylor basis (degree <= 2)
- Quadrature: FPolynomialFill consistency check

[test_QuadratureCachePeer.cpp](../../test/cpp/Geom/test_QuadratureCachePeer.cpp)


## `solver_test_aggregation_amg`

| Level | CTest registrations |
|---|---|
| 0 | `solver_aggregation_amg` |

[test_AggregationAMG.cpp](../../test/cpp/Solver/test_AggregationAMG.cpp)

- A reused aggregation hierarchy matches a dense two-level block cycle
- Block Jacobi preserves resolvable modes with a structural zero diagonal

## `solver_test_direct`

| Level | CTest registrations |
|---|---|
| 0 | `solver_direct` |

[test_Direct.cpp](../../test/cpp/Solver/test_Direct.cpp)

- Direct 2D periodic: symbolic factorization has fill-in
- Direct symbolic factorization ignores self edges
- Direct 2D periodic: MatMul is correct
- Direct 2D periodic: complete LU solve is exact
- Direct 2D periodic: ILU(0) is approximate but reduces residual
- Direct 2D periodic: ILU(1) converges faster than ILU(0) as preconditioner
- Direct 2D periodic LDLT: MatMul matches LU MatMul for symmetric system
- Direct 2D periodic LDLT: complete decompose + solve is exact
- Direct core ordering is stable within local parts and precedes symbolic fill
- ILU zero-fill support does not bound upper triangular readiness
- Generic ranged factors and separate triangular outputs preserve the full solve
- Rebuilding symbolic factors replaces previous row incidences
- Unreported symbolic construction is rank local
- Native block matrix shares immutable row metadata and retains access lifetime
- Fixed and dynamic diagonal inverse retain the guarded LU and singular SVD policies
- Owned logical table recognition retains pitch and rejects changed component ordering
- Native small-block LU specializations solve coupled blocks on all uniform layouts
- Native block LU solves strided DOFs and rejects stale matrices and aliases
- Native block LU supports empty owners and counts guarded inverse fallback
- Native triangular phases retain owners and advance independent local parts

## `solver_test_fas`

| Level | CTest registrations |
|---|---|
| 0 | `solver_fas` |

[test_FAS.cpp](../../test/cpp/Solver/test_FAS.cpp)

- FAS distinguishes state and residual restriction and preserves incoming references through three levels
- Identity FAS cancels a common large linear term before forming tau
- Nonlinear FAS coarse forcing preserves a known fine solution
- FAS rejects ambiguous ownership controls aliases and stateful operator callbacks
- Selected FAS solves principal equations and freezes other components across strided levels
- Selected FAS rejects invalid selections and callbacks that write frozen means

## `solver_test_field_gmres`

| Level | CTest registrations |
|---|---|
| 0 | `solver_field_gmres` |

[test_FieldGMRES.cpp](../../test/cpp/Solver/test_FieldGMRES.cpp)

- Field GMRES handles first-vector closure true residuals and extreme fixed preconditioner scaling
- Field GMRES restarts nonnormal systems across layouts with empty local ranks
- Field GMRES reports algebraic failure without false convergence

## `solver_test_field_time_stepper`

| Level | CTest registrations |
|---|---|
| 0 | `solver_field_time_stepper` |

[test_FieldTimeStepper.cpp](../../test/cpp/Solver/test_FieldTimeStepper.cpp)

- Time methods own only their relevant history and stage buffers
- DITR transformed algebraic rate equations have independent midpoint and endpoint rows
- Public DITR method preserves Euler one-correction sequence and I plus H stopping
- Public normalized BDF sources exclude the current unknown and use accepted step ratios
- Native implicit time methods converge with independent nonautonomous exact stages
- Time group tickets reject foreign sessions and reused scratch generations
- External controller interleaves coherent stages without changing native time equations
- Serialized BDF and DIRK stages restore their trial source and completed rate prefix
- Native BDF1 steady limit has zero physical mass and history source at a frozen time
- Native time target comparison excludes only relative roundoff remainders
- Native BDF history is committed explicitly and retries use the last accepted step ratio
- Public DITR converges to the independent original two-equation dense oracle
- Public DITR retains nonautonomous temporal order and controller-owned capped retry
- DITR rejects rebound intermediate and stateful rate callbacks without touching accepted state
- Time facade reserves DITR publication while the external controller owns convergence
- Time facade restores DITR coupled state at a completed iteration boundary

## `cfv_test_characteristic_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `cfv_characteristic_cuda_np1` |

[test_CharacteristicCUDA.cu](../../test/cpp/CFV/test_CharacteristicCUDA.cu)

- Queued matrix production validates once before modal WBAP without a host drain
- Queued gradient WBAP shares the producer validation contract and leaves scalar components scalar

## `cfv_test_device_transferable`

| Level | CTest registrations |
|---|---|
| 1 | `cfv_device_transferable_np1` |

[test_DeviceTransferable.cu](../../test/cpp/CFV/test_DeviceTransferable.cu)


## `cfv_test_fv_completion`

| Level | CTest registrations |
|---|---|
| 1 | `cfv_fv_completion_np1`, `cfv_fv_completion_np2` |
| 2 | `cfv_fv_completion_np4`, `cfv_fv_completion_np8` |

[test_FVCompletion.cpp](../../test/cpp/CFV/test_FVCompletion.cpp)

- Rotated periodic sector preserves vector linear fields across distributed ownership
- Axisymmetric metrics and all gradient methods obey radial linear moments at the axis
- Raw reconstruction operators preserve symmetry and agree with a preconditioned Jacobi step
- Cached owned AoS polynomial samples match strided output for common equation widths
- Shared basis policies snapshot points and sample cubic values and derivatives
- Shared quadrature face banks preserve periodic parent frames and scattered outputs
- Bundled polynomial values and gradients preserve the independent sampler
- Existing GMRES consumes native FV operators and DNDS field algebra through an explicit vector adapter

## `cfv_test_reconstruction_capabilities`

| Level | CTest registrations |
|---|---|
| 1 | `cfv_reconstruction_capabilities_np1`, `cfv_reconstruction_capabilities_np2` |
| 2 | `cfv_reconstruction_capabilities_np4`, `cfv_reconstruction_capabilities_np8` |

[test_ReconstructionCapabilities.cpp](../../test/cpp/CFV/test_ReconstructionCapabilities.cpp)

- Sweep capability profiles preserve full Jacobi SOR and WBAP arithmetic in 2D and 3D
- Original operator and preconditioner profiles remain independent under complete SVD truncation
- Inverse-only preparation includes self-periodic diagonal corrections
- Sampling-only preparation is independent of matrix capabilities and ranks agree before setup

## `dnds_test_array_derived`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_array_derived_np1`, `dnds_array_derived_np2` |
| 2 | `dnds_array_derived_np4`, `dnds_array_derived_np8` |

[test_ArrayDerived.cpp](../../test/cpp/DNDS/test_ArrayDerived.cpp)

- Audit regression: cloned matrix row shapes detach on resize
- ArrayAdjacency basics
- ArrayAdjacency ghost communication
- ArrayEigenVector basics
- ArrayEigenVector ghost communication
- ArrayEigenMatrix basics
- ArrayEigenMatrix dynamic sizes
- ArrayEigenMatrix ghost communication
- ArrayEigenMatrixBatch basics
- ArrayEigenUniMatrixBatch basics
- ArrayEigenUniMatrixBatch static sizes
- ArrayAdjacency clone
- ArrayEigenMatrix NonUniformSize rows
- ArrayEigenVector static size
- ArrayEigenMatrixBatch ghost communication
- ArrayAdjacency fixed-size variant
- ArrayAdjacency parametric
- ArrayEigenVector parametric
- ArrayEigenMatrix parametric

## `dnds_test_array_dof`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_array_dof_np1`, `dnds_array_dof_np2` |
| 2 | `dnds_array_dof_np4`, `dnds_array_dof_np8` |

[test_ArrayDOF.cpp](../../test/cpp/DNDS/test_ArrayDOF.cpp)

- Audit regression: difference norm retains shape on empty ranks
- ArrayDOF setConstant with Eigen matrix
- ArrayDOF operator+= array and matrix
- ArrayDOF operator-= and operator*=
- ArrayDOF addTo
- ArrayDOF norm2
- ArrayDOF min and max
- ArrayDOF sum
- ArrayDOF componentWiseNorm1
- ArrayDOF componentWiseNorm1 difference
- ArrayDOF element-wise multiply and divide
- ArrayDOF operator= (copy values)
- ArrayDOF clone
- ArrayDOF scalar array multiply
- ArrayDOF norm2 zero array
- ArrayDOF dot self equals norm2 squared
- ArrayDOF parametric over nVars

## `dnds_test_array_dof_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_array_dof_cuda_np1`, `dnds_array_dof_cuda_np2` |
| 2 | `dnds_array_dof_cuda_np4`, `dnds_array_dof_cuda_np8` |

[test_ArrayDOF_CUDA.cu](../../test/cpp/DNDS/test_ArrayDOF_CUDA.cu)

- CUDA boundary: uniform element matrix operations
- CUDA boundary: uniform scalar row multiplication
- CUDA boundary: CSR scalar row multiplication
- CUDA ownership: leases retain device storage after replacement

## `dnds_test_array_transformer`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_array_transformer_np1`, `dnds_array_transformer_np2` |
| 2 | `dnds_array_transformer_np4`, `dnds_array_transformer_np8` |

[test_ArrayTransformer.cpp](../../test/cpp/DNDS/test_ArrayTransformer.cpp)

- Audit batch 2: in-situ packing preserves padded rows and push lifecycle
- Audit batch 2: empty in-situ push is repeatable
- Audit regression: transformer copies preserve backend
- Audit regression: reinitialize partially initialized transformer
- ParArray basics
- ArrayTransformer pull - TABLE_StaticFixed
- ArrayTransformer pull - TABLE_Fixed (DynamicSize)
- ArrayTransformer pull - CSR layout
- ArrayTransformer pull - std::array elements
- ArrayTransformer persistent pull
- ArrayTransformer BorrowGGIndexing
- ArrayTransformer push
- ArrayTransformer pull

## `dnds_test_cauchy_auxiliary_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_auxiliary_cuda` |

[test_AuxiliaryCUDA.cu](../../test/cpp/Cauchy/test_AuxiliaryCUDA.cu)

- CUDA boundary producers match generic CPU face exterior and interface states
- CUDA auxiliary means match generic CPU thermodynamics and RANS transport with ghost gradients
- CUDA face auxiliaries distinguish rotated neighbor physical boundary and interface then reduce by cell

## `dnds_test_cauchy_auxiliary_preparation`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_auxiliary_preparation_np1`, `dnds_cauchy_auxiliary_preparation_np2` |
| 2 | `dnds_cauchy_auxiliary_preparation_np4`, `dnds_cauchy_auxiliary_preparation_np8` |

[test_AuxiliaryPreparation.cpp](../../test/cpp/Cauchy/test_AuxiliaryPreparation.cpp)

- Auxiliary core completes during a real mean halo write and full publication waits for both phases
- Queued CPU auxiliary accepts pending means and feeds consumers before publication
- High-order logical auxiliary maxima match the full pointwise flow path
- Auxiliary abandonment rebind and phase errors never publish a complete result
- Unused private face scratch is safe when caller supplied outputs move to a larger face-only problem
- Auxiliary reductions count both incidences of a self-periodic face
- Auxiliary transaction retains input output and geometry when original handles expire
- Auxiliary provenance includes DOF row identity when payload buffers are unchanged
- Synchronous residual completes its queued core before dependent rejection unwinds leases
- FV uses the same queued halo dependency sequence with inline and worker launchers

## `dnds_test_cauchy_characteristics_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_characteristics_cuda` |

[test_CharacteristicsCUDA.cu](../../test/cpp/Cauchy/test_CharacteristicsCUDA.cu)

- CUDA analytic mean characteristics match CPU inverse and canonical periodic face frames

## `dnds_test_cauchy_derived_fields`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_derived_fields` |

[test_DerivedFields.cpp](../../test/cpp/Cauchy/test_DerivedFields.cpp)

- Derived point refresh fills selected properties with lazy provenance and stale temperature guesses
- Mixture derived refresh consumes compatible stale temperatures across coupled and separate layouts

## `dnds_test_cauchy_finite_volume`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_finite_volume_np1`, `dnds_cauchy_finite_volume_np2` |
| 2 | `dnds_cauchy_finite_volume_np4`, `dnds_cauchy_finite_volume_np8` |

[test_FiniteVolume.cpp](../../test/cpp/Cauchy/test_FiniteVolume.cpp)

- Finite-volume workspace identity follows its shared plan and rejects foreign or expired plans
- Finalized reconstruction aliases unfiltered storage and retains publication provenance
- One finite-volume scratch workspace serves independent durable reconstruction states
- Default Cauchy CGNS loader propagates the configured preflight policy
- Cauchy collective switch reaches prepared halos without changing the P3 RHS
- Semantic finite-volume dispatch preserves the trusted flow kernel and rejects an incompatible model before communication
- Euler reconstruction PP selects a mean or progressively truncated reference across layouts during halo exchange
- WBAP and PP flux filtering do not overwrite the unlimited VR warm guess
- Auxiliary provenance and persistent workspaces reject a different row view of the same allocation
- Cauchy communication worker preserves nonconstant P3 RHS
- Cauchy P0 explicit P1 and degree-one VR retain their distinct storage across layouts and workers
- Cauchy direct P1 owns Barth and thermal PP without variational storage
- Native viscous Cauchy residual reproduces linear Couette heat production
- Cauchy SSPRK3 preserves periodic constants and rolls back rejected trials
- Cauchy periodic density transport has independently measured second-order spatial residual
- Cauchy periodic transport SSPRK3 advances a smooth physical state and conserves mass
- Every Cauchy Riemann selection is integrated into periodic SSPRK3
- Cauchy variational reconstruction with WBAP advances a nonconstant conservative field
- Cauchy physical execution includes ranks with empty owned cells
- Cauchy rejects inadmissible trial collectively and retries without accepted-state corruption
- Cauchy matching-partition restart preserves layouts continuation and guards mesh identity
- Cauchy real single-domain Simulation owns its accepted clock and restart epoch
- Cauchy P3 non-PP vortex cell-average evolution converges and conserves on distributed torus
- Cauchy P3 characteristic 3WBAP advances shock data with inline and worker launchers
- Cauchy observation cadence leaves a fixed-sweep P3 trajectory unchanged
- Cauchy P3 boundary polynomial projection preserves an exact linear extrapolated flow
- Cauchy P3 thermal PP preserves means and global conservation with forced temperature constraints
- Inactive Cauchy P3 thermal PP leaves the full numerical trajectory bitwise unchanged
- Cauchy P3 fixed-sweep restart preserves the polynomial warm start and exact continuation
- Cauchy thermal PP rejects invalid low-order stages collectively without clipping and permits retry
- Canonical flow5 positivity workspace produces the established high and low residuals without the archived FV frontend
- Cauchy repeated parallel VTKHDF output preserves topology and nonconstant fields with explicit compression
- Riemann spatial stencils are computed by owners and exchanged across periodic partitions
- Stateless auxiliary preparation supports independently optional face and cell outputs
- Auxiliary reuse rejects active writers changed inputs replaced outputs and invalid shapes
- Mean boundary spectra use stage kinematics while analytic RHS uses boundary stage time
- A checkpoint predating spatial Riemann options retains explicit manual behavior
- Checkpoint PP policy preserves pre-Euler archives without silently changing active reconstruction limiting
- A rank-local Riemann stencil setting rejects collectively before optional halo construction
- Optional derived auxiliaries refresh owned rows during halo exchange and feed the stateless RHS
- Cauchy P3 viscous WBAP agrees with shared first-order, full and uncached basis
- Cauchy overlap and unsplit orchestration agree across CPU launchers
- Cauchy P1 characteristic WBAP publishes direct physical gradients
- Primitive gradient overlap preserves conservative reconstruction across layouts and CPU launchers
- Stateless RHS consumes packed auxiliary views and preserves frozen inputs
- Direct FV consumes physical gradients without coefficients and splits flux from source
- RHS checked and production-fatal policies share normal arithmetic

## `dnds_test_cauchy_foundation`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_foundation_np1`, `dnds_cauchy_foundation_np2` |
| 2 | `dnds_cauchy_foundation_np4`, `dnds_cauchy_foundation_np8` |

[test_Foundation.cpp](../../test/cpp/Cauchy/test_Foundation.cpp)

- Cauchy configuration preflight coordinates invalid or differing rank inputs
- Domain execution contexts retain named lanes and backend lifetime
- Prepared operator records retain their owners through execution retirement
- Imported interface completions progress on the controller and retain receiver resources
- Cauchy domain placement preserves rank namespaces and isolates shared groups
- Frozen trace rounds give identical shared and disjoint rank numerical schemes
- Named coupled DOF blocks preserve layout independence subset algebra and owned norms

## `dnds_test_cauchy_frozen_jacobian_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_frozen_jacobian_cuda` |

[test_FrozenJacobianCUDA.cu](../../test/cpp/Cauchy/test_FrozenJacobianCUDA.cu)

- CUDA frozen Jacobians match CPU blocks and actions including ALE periodic projection and RANS stress
- CUDA diagonal initialization preserves CPU mass and moving-frame block convention
- CUDA centered face diagonal matches CPU wall chains Roe matrices and self periodic contribution
- managed frozen Jacobian retains real periodic mesh geometry through queued application

## `dnds_test_cauchy_gradient_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_gradient_cuda` |

[test_GradientCUDA.cu](../../test/cpp/Cauchy/test_GradientCUDA.cu)

- CUDA Cauchy centered Barth PP and primitive means match generic CPU

## `dnds_test_cauchy_interface_iteration`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_interface_iteration_np1`, `dnds_cauchy_interface_iteration_np2` |
| 2 | `dnds_cauchy_interface_iteration_np4`, `dnds_cauchy_interface_iteration_np8` |

[test_InterfaceIteration.cpp](../../test/cpp/Cauchy/test_InterfaceIteration.cpp)

- Production single-stage policy preserves frozen/updated traces and real overlap
- Retained packet reads reject refresh before posting and preserve received values
- Coordinated DITR policy retains idle ranks and midpoint-before-endpoint ordering

## `dnds_test_cauchy_limited_vr`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_limited_vr_np1`, `dnds_cauchy_limited_vr_np2` |
| 2 | `dnds_cauchy_limited_vr_np4`, `dnds_cauchy_limited_vr_np8` |

[test_LimitedVR.cpp](../../test/cpp/Cauchy/test_LimitedVR.cpp)

- LVR variants expand defaults per input layer with explicit overrides and strict validation
- Native Cauchy LVR prepares active v2/v3 targets identically across layouts and CPU schedules
- Native Cauchy LVR expansion admission changes the shock gate while constant fields stay unlimited
- Canonical EquationDofs LVR retains the established target gates and residual
- LVR layer or capability differences reject before building divergent transfer bindings
- Zero LVR caps recover the unlimited operator and omit unnecessary gate work
- LVR history errors preserve accepted means and replace failed reconstruction output
- LVR periodic mean and target halos agree across split and unsplit CPU execution

## `dnds_test_cauchy_limited_vr_checkpoint`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_limited_vr_checkpoint_np1`, `dnds_cauchy_limited_vr_checkpoint_np2`, `dnds_cauchy_limited_vr_checkpoint_np4` |
| 2 | `dnds_cauchy_limited_vr_checkpoint_np8` |

[test_LimitedVRCheckpoint.cpp](../../test/cpp/Cauchy/test_LimitedVRCheckpoint.cpp)

- LVR warm restart and output cadence preserve the complete implicit trajectory
- LVR checkpoint remaps warm history under local permutation and cold-starts a new degree
- LVR warm restart redistributes nonzero history from fewer writers into reordered strided fields
- LVR fine reconstruction connects to limited-gradient and constant polynomial FAS levels

## `dnds_test_cauchy_limited_vr_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_limited_vr_cuda` |

[test_LimitedVRCUDA.cu](../../test/cpp/Cauchy/test_LimitedVRCUDA.cu)

- LVR center target guard and direct sensors preserve CPU CUDA branch gates

## `dnds_test_cauchy_linear_block_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_linear_block_cuda` |

[test_LinearBlockCUDA.cpp](../../test/cpp/Cauchy/test_LinearBlockCUDA.cpp)

- CUDA assembled actions and Jacobi preserve sparse block order and core halo separation

## `dnds_test_cauchy_linear_inverse_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_linear_inverse_cuda` |

[LinearInverseReference.cpp](../../test/cpp/Cauchy/LinearInverseReference.cpp)


[test_LinearInverseCUDA.cu](../../test/cpp/Cauchy/test_LinearInverseCUDA.cu)

- Queued CUDA diagonal SVD and scalar inverses match native CPU actions
- Optional CUDA inverse refinement preserves the independent SST rhoK action
- CUDA inverse observation rejects invalid diagonal before Jacobi publication
- Fixed small LU condition guard matches Eigen on pivoted and near-singular blocks
- Guarded CUDA SVD fallback matches CPU for weak and discarded modes
- Queued inverses preserve independent diagonal and inverse matrix layouts

## `dnds_test_cauchy_linear_solver_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_linear_solver_cuda`, `dnds_cauchy_linear_solver_cuda_np2` |

[test_LinearSolverCUDA.cu](../../test/cpp/Cauchy/test_LinearSolverCUDA.cu)

- Reserved weighted products and finite checks use deterministic resident reductions
- Distributed coupled Jacobi and left GMRES share queued orchestration
- Copy-free local AmgX consumes native int32 unsorted row-major blocks

## `dnds_test_cauchy_multidomain_mesh`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_multidomain_mesh_np1`, `dnds_cauchy_multidomain_mesh_np2` |
| 2 | `dnds_cauchy_multidomain_mesh_np4`, `dnds_cauchy_multidomain_mesh_np8` |

[test_MultidomainMesh.cpp](../../test/cpp/Cauchy/test_MultidomainMesh.cpp)

- P1 Delaunay disk and periodic box-hole fixtures have complementary area and moving boundary matches

## `dnds_test_cauchy_nonlinear`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_nonlinear_np1`, `dnds_cauchy_nonlinear_np2` |
| 2 | `dnds_cauchy_nonlinear_np4`, `dnds_cauchy_nonlinear_np8` |

[test_NonlinearSolver.cpp](../../test/cpp/Cauchy/test_NonlinearSolver.cpp)

- Cell geometry reduces pseudo-time in larger cells across periodic and MPI faces
- Update policies register without changing nonlinear orchestration
- Nonlinear controller state has a versioned JSON round trip
- Mean-state Krylov scaling uses volume averages and one common momentum scale
- Shared nonlinear controller observes coupled updates once and advances CFL once
- Nonlinear controller resumes after a completed correction without resetting references or CFL
- Cauchy nonlinear progress reports completed iterations and zero-based stage identity
- Implicit convergence retains initial and running references without hiding unselected components
- Distributed implicit L1 L2 norms and selected-component stops use the actual manufactured defect
- One correction reuses the ordinary solve update without a stopping loop
- Euler alpha reductions agree sparse triggers and reject mismatched policies or statistics
- Global Euler RHS alpha policies affect unmarked rows only for minimumAll in RHS and dTau styles
- Native implicit stage recovers a constant manufactured solution from distributed nonconstant means
- Implicit stage cap reports unconverged and sessions exclude concurrent solves
- Implicit bad corrections reject without poisoning state and nonlinear controls agree before allocation
- Subset correction masks before Euler PP and preserves frozen means exactly
- Optional nonlinear phase output preserves exact corrections and records early failures without extra collectives

## `dnds_test_cauchy_physical_timestep`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_physical_timestep_np1`, `dnds_cauchy_physical_timestep_np2` |
| 2 | `dnds_cauchy_physical_timestep_np4`, `dnds_cauchy_physical_timestep_np8` |

[test_PhysicalTimeStep.cpp](../../test/cpp/Cauchy/test_PhysicalTimeStep.cpp)

- An explicit initial stage rejects a finite state with nonzero discrete continuity
- Physical timestep PP agrees with Euler global compress-one bounds across layouts and active halos
- Equation mean trial and thermal dt preserve all block layouts and the ordinary warm guess
- Typed mean residual skips VR and filtering without modifying the normal warm guess
- Domain timestep growth follows committed dt through prepare discard and blocked commit

## `dnds_test_cauchy_physics`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_physics` |

[test_PerfectGas.cpp](../../test/cpp/Cauchy/test_PerfectGas.cpp)

- QCR stress and energy work match an independent commutator and legacy Euler
- Equation thermal view preserves the established perfect-gas increment and quadrature rules
- Polynomial viscous flux reproduces Euler conservative gradient and face-jump formulas
- Quadrature reconstruction compression preserves Euler common fractions and energy references
- Implicit energy compression preserves Euler scheme-one mathematics with explicit energy references
- Implicit flow compression enforces density and both temperature and energy constraints
- Equation-owned physical flux Jacobian differentiates all three momentum components
- Physical boundary Jacobians match conservative exterior-state derivatives
- Thermal PP uses an internal-energy temperature bound including shifted energy references
- Cold mean limiter characteristics retain a reliable inverse without altering physical sound speed
- Perfect-gas characteristic matrices match independent Euler analytic matrices in rotated frames
- Cached generic characteristics preserve caloric energy offsets and the frozen EOS Jacobian
- Cauchy prescribed DMR shock and boundary stage times match canonical conservative data
- Cauchy beta-five vortex oracle advects periodically and preserves isentropy
- Cauchy farfield respects supersonic directions and subsonic incoming component count
- Flow5 retains transverse momentum on planar geometry and exact physical flux
- Inviscid kernels agree with legacy flux and conserve under oriented face reversal
- Newtonian Fourier kernels reproduce analytic shear heat flux and legacy three-dimensional stress
- Boundary values and exterior states implement slip no-slip and thermal wall contracts
- Farfield sonic branches and approximate wall Jacobians follow Euler conventions
- Perfect gas transport and flux preparation preserve typed local arithmetic
- Reduced perfect gas flux preserves shock and low pressure contact bounds

## `dnds_test_cauchy_point_physics_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_point_physics_cuda` |

[test_PointPhysics_CUDA.cu](../../test/cpp/Cauchy/test_PointPhysics_CUDA.cu)

- Shared perfect-gas Riemann arithmetic agrees on CUDA for all schemes with distinct means and moving faces
- SA and k-omega CUDA point closures agree with canonical CPU source transport and diffusion
- Perfect-gas CUDA boundaries preserve physical and exterior targets in a moving frame

## `dnds_test_cauchy_positivity_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_positivity_cuda` |

[test_PositivityCUDA.cu](../../test/cpp/Cauchy/test_PositivityCUDA.cu)

- CUDA physical-dt forecast ignores RANS bounds and reports invalid means
- CUDA polynomial positivity matches CPU flow compression and independent RANS repair

## `dnds_test_cauchy_progress_output`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_progress_output` |

[test_ProgressOutput.cpp](../../test/cpp/Cauchy/test_ProgressOutput.cpp)

- Output queue snapshots precede later mutations and writes stay ordered
- Output queue records a terminal writer failure without running later work
- One progress controller constructs iteration stage and step CSV rows
- Steady ACM SST progress groups console residuals and displays an infinite physical step
- Resolved output plan centralizes names cadence inheritance and exact endpoint snapping

## `dnds_test_cauchy_rans_reconstruction_reduced`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_rans_reconstruction_reduced` |

[test_RANSReconstructionReduced.cpp](../../test/cpp/Cauchy/test_RANSReconstructionReduced.cpp)

- Reduced-order SA reconstruction compresses its scalar about the mean

## `dnds_test_cauchy_rans_state`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_rans_state` |

[test_EquationRANS.cpp](../../test/cpp/Cauchy/test_EquationRANS.cpp)

- Generic wall Jacobian follows the centralized RANS exterior-state law
- RANS model and point gathering follow logical blocks across split and coupled layouts
- RANS scalar repairs retain the Euler componentwise and exponential rules
- RANS reconstruction scales about the mean with a zero scalar floor
- Named RANS initialization and numerical reference resolve against the model
- RANS no-slip targets retain conservative density and viscosity conventions

## `dnds_test_cauchy_rhs_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_rhs_cuda` |

[test_RHSCUDA.cu](../../test/cpp/Cauchy/test_RHSCUDA.cu)

- CUDA RHS parameter publication rejects species in the perfect-gas profile
- CUDA integrated RHS matches generic CPU with periodic halo wall and interface incidences
- CUDA canonical ghost-left and self-periodic faces preserve owned signs and incidence frames

## `dnds_test_cauchy_rhs_face_packages_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_rhs_face_packages_cuda` |

[test_RHSFacePackagesCUDA.cu](../../test/cpp/Cauchy/test_RHSFacePackagesCUDA.cu)

- Complete-face packages preserve mixed rules tails and bounded point tiles
- Nine-point packages pack tightly across warps within the CTA budget

## `dnds_test_cauchy_rigid_motion`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_rigid_motion` |

[test_RigidMotion.cpp](../../test/cpp/Cauchy/test_RigidMotion.cpp)

- Rigid frames preserve absolute energy and provide consistent mesh velocity and rotation derivatives

## `dnds_test_cauchy_source_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_source_cuda` |

[test_SourceCUDA.cu](../../test/cpp/Cauchy/test_SourceCUDA.cu)

- CUDA RANS quadrature sources and destruction diagonals match generic CPU
- CUDA moving-axis source preserves scalar components

## `dnds_test_cauchy_state_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_state_cuda` |

[test_StateCUDA.cu](../../test/cpp/Cauchy/test_StateCUDA.cu)

- Queued limiter statistics keep integer counts and deterministic minima
- Queued status reduction preserves earlier failures across reused producer buffers
- Spatial placement retains metric identity and allocates VR boundary values without host staging
- Queued boundary operator retains P3 basis and matches CPU physical and mirrored targets
- Managed CUDA volume sources retain distributed geometry and match CPU sampling
- SST mean-gradient auxiliary schedule is shared by inline worker and CUDA execution
- Queued auxiliary operator matches CPU on a distributed periodic mesh with optional outputs
- Managed CUDA flux matches generic CPU for P0 P1 and P3 with distributed inputs
- Queued CUDA owned-mean and CFL reductions match CPU including empty partitions and first failure
- CUDA analytic error uses the CPU oracle quadrature and fixed-tree rank-local reductions
- Managed characteristics feed WBAP without a host observation between operators

## `dnds_test_cauchy_wall_distance`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_cauchy_wall_distance_np1`, `dnds_cauchy_wall_distance_np2` |
| 2 | `dnds_cauchy_wall_distance_np4`, `dnds_cauchy_wall_distance_np8` |

[test_WallDistanceSamples.cpp](../../test/cpp/Cauchy/test_WallDistanceSamples.cpp)

- Nodal wall sampling retains P1 and P2 shapes on distributed cells and faces
- First-layer wall distance selects a cell value for volume and mean samples

## `dnds_test_config`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_config` |

[test_Config.cpp](../../test/cpp/DNDS/test_Config.cpp)

- Audit batch 2: config integer conversion rejects invalid values
- Typed config schemas validate complete inputs and layered partial patches
- Audit batch 2: registration publishes only complete metadata
- Audit batch 2: failed registration retries without partial fields
- Config registration rejects same-type recursion

## `dnds_test_csv_log`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_csv_log` |

[test_CsvLog.cpp](../../test/cpp/DNDS/test_CsvLog.cpp)

- Typed CSV rows preserve schema types blanks escaping and lazy rotation

## `dnds_test_hard_eigen`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_hard_eigen` |

[test_HardEigen.cpp](../../test/cpp/DNDS/test_HardEigen.cpp)

- Degenerate inertia frames use coordinate axes and deterministic principal planes
- SVD cutoff retains constrained modes and reports the actual rank
- SVD diagnostics match automatic Eigen cutoff and rank-zero matrices
- Explicit SVD cutoffs are scale relative with optional absolute threshold

## `dnds_test_index_mapping`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_index_mapping_np1`, `dnds_index_mapping_np2` |
| 2 | `dnds_index_mapping_np4`, `dnds_index_mapping_np8` |

[test_IndexMapping.cpp](../../test/cpp/DNDS/test_IndexMapping.cpp)

- GlobalOffsetsMapping uniform distribution
- GlobalOffsetsMapping non-uniform distribution
- GlobalOffsetsMapping search
- OffsetAscendIndexMapping pull-based construction
- OffsetAscendIndexMapping search_indexAppend
- OffsetAscendIndexMapping empty ghost set

## `dnds_test_legacy_array_bridge`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_legacy_array_bridge` |

[test_LegacyArrayBridge.cpp](../../test/cpp/DNDS/test_LegacyArrayBridge.cpp)

- Legacy bridge reuses controller and permits scoped edits both ways
- Legacy bridge retains exact allocations through replacement and deep copies
- Legacy bridge converts CSR offsets and retains published row metadata
- Legacy bridge snapshots empty publication and escaped padded row sizes
- Legacy bridge converts static and dynamic padded lengths
- Legacy bridge validates actual shape byte range type schema and host placement
- Legacy bridge rejects malformed CSR publication and retains native subspan offsets
- Legacy bridge supports a registered POD MPI extent without changing element identity

## `dnds_test_mpi`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_mpi_np1`, `dnds_mpi_np2` |
| 2 | `dnds_mpi_np4`, `dnds_mpi_np8` |

[test_MPI.cpp](../../test/cpp/DNDS/test_MPI.cpp)

- MPIInfo basics
- AgreeOptionalInteger is exact across automatic and explicit policies
- MPI Allreduce
- MPI Scan
- MPI Allgather
- MPI Bcast
- MPI Barrier
- MPI BasicType_To_MPIIntType
- CommStrategy
- MPI Alltoall
- Parametric Allreduce MPI_SUM
- Parametric Allgather rank values

## `dnds_test_native_dof_algebra_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_native_dof_algebra_cuda` |

[test_NativeDofAlgebra_CUDA.cu](../../test/cpp/DNDS/test_NativeDofAlgebra_CUDA.cu)

- CUDA operator input cache orders queued conversion before its consumer
- Native CUDA algebra retains float/double fields, joins streams and excludes padding/ghosts

## `dnds_test_object_contracts`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_object_contracts_np1`, `dnds_object_contracts_np2` |
| 2 | `dnds_object_contracts_np4`, `dnds_object_contracts_np8` |

[test_LocalChecks.cpp](../../test/cpp/DNDS/test_LocalChecks.cpp)

- Local checks: rejected static width preserves the array
- Local checks: padded rows reject negative widths
- Local checks: batch shape sentinel is exactly minus one
- Local checks: size arithmetic rejects limits without allocations
- Local checks: matrix products and dimensions are checked first
- Local checks: gathered mapping handles empty ranks and invalid input
- Local checks: redistribution rejects rank-local invalid layouts collectively
- Local checks: redistribution preserves exact sparse and high ID row order
- Local checks: high ID redistribution faults reject collectively and allow retry
- Local checks: H5 redistribution preserves sparse and high IDs across ownership changes
- Local checks: serialized payload must match array shape
- Local checks: JSON CSR terminal and padded widths match payload
- Local checks: zero width H5 rows preserve their explicit slice
- Local timing: length broadcasts versus checked allgather

[test_ObjectContracts.cpp](../../test/cpp/DNDS/test_ObjectContracts.cpp)

- Audit batch 2: moved vectors remain reusable
- Audit batch 2: moved arrays reset structural state
- Audit batch 2: swaps reject unequal row and matrix shapes
- Audit batch 2: pair swaps validate both arrays before mutation

## `dnds_test_permutation_transfer`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_permutation_transfer_np1`, `dnds_permutation_transfer_np2` |
| 2 | `dnds_permutation_transfer_np4`, `dnds_permutation_transfer_np8` |

[test_PermutationTransfer.cpp](../../test/cpp/DNDS/test_PermutationTransfer.cpp)

- PermutationTransfer::fromLocalPermutation reverse
- PermutationTransfer::fromLocalPermutation identity
- PermutationTransfer::fromPartition all-local
- PermutationTransfer::fromPartition round-robin
- PermutationTransfer::buildLookup resolve
- PermutationTransfer distributed lookup cross-rank
- PermutationTransfer::transferRows CSR local permutation

## `dnds_test_runtime`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_runtime` |

[test_Runtime.cpp](../../test/cpp/DNDS/test_Runtime.cpp)

- OptionalRef preserves constness and checked synchronous rebinding
- Final allocation ownership passes only the actual final release context to its provider
- Concurrent contextual releases never free a shared allocation more than once
- Async Array construction releases a rejected provider allocation with its explicit context
- Allocator policy and pooled storage outlive asynchronous access owners
- Field keeps association and distribution outside allocation
- Mutation versions invalidate selected dependencies and fail closed
- Shallow mesh controller aliases share exclusion and invalidation
- Runtime aliases share access state and versions, copies do not
- Runtime access owns allocation after handle replacement
- Legacy Array shares native host allocation without changing row API

## `dnds_test_runtime_array_io`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_runtime_array_io_np1`, `dnds_runtime_array_io_np2` |
| 2 | `dnds_runtime_array_io_np4`, `dnds_runtime_array_io_np8` |

[test_RuntimeArrayIO.cpp](../../test/cpp/DNDS/test_RuntimeArrayIO.cpp)

- Native parallel checkpoint matching partitions and direct read
- Parallel field bundle restores uneven reversed partitions with variable row lengths
- Native JSON checkpoint padded rows and POD schema
- Native physical table archives restore placement and read directly into other layouts
- Per-rank JSON physical tables retain logical values through direct format conversion

## `dnds_test_runtime_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_runtime_cuda` |

[test_Runtime_CUDA.cu](../../test/cpp/DNDS/test_Runtime_CUDA.cu)

- Pinned host snapshots retain DMA leases and are accounted once
- Backend metadata spans survive asynchronous launch after handles are replaced
- CUDA five layout policies static width stride and capacity reach actual kernels
- CUDA eager submissions preserve all five layout policies in launcher and device views
- CUDA array allocation accounting includes host and device row metadata without alias duplication
- Native ragged Array CUDA views retain metadata through replacement
- Padded CUDA views use logical row sizes and preserve capacity padding
- Fixed POD CUDA rows use sized owned and combined read spans
- External CUDA allocation and shared CSR metadata survive pending kernels
- Runtime CUDA residency, cross-stream transfer and retained leases
- Copied CUDA completions share terminal observation without releasing attached leases
- Legacy device mirror cannot supply stale native host access
- Runtime CUDA destructor fences an unsubmitted launch
- CUDA column and tiled row views preserve physical padding and static strides
- CUDA explicit format conversion never reads device row metadata on the host

## `dnds_test_runtime_exchange`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_runtime_exchange_np1`, `dnds_runtime_exchange_np2` |
| 2 | `dnds_runtime_exchange_np4`, `dnds_runtime_exchange_np8` |

[test_RuntimeExchange.cpp](../../test/cpp/DNDS/test_RuntimeExchange.cpp)

- Exchange routes arbitrary rows and transpose sums, retaining buffers
- Exchange collective validation and ranks with no owner rows

## `dnds_test_serializer`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_serializer_np1`, `dnds_serializer_np2` |
| 2 | `dnds_serializer_np4`, `dnds_serializer_np8` |

[test_Serializer.cpp](../../test/cpp/DNDS/test_Serializer.cpp)

- Audit batch 2: shared read cache owns output independently
- Audit batch 2: HDF5 shared reads respect regions and offsets
- Audit batch 2: repeated distributed CSR reads retain global offsets
- CSR sharing: array round trip retains shared structure
- CSR sharing: explicit API separates raw values and rejects changed slices
- CSR sharing: empty ranks retain their terminal offset
- CSR sharing: mixed writer identities do not create inconsistent references
- CSR sharing: reads legacy raw-vector row-start encoding
- SerializerJSON scalar round-trip
- ArrayPair redistribute — same np round-trip
- ArrayPair redistribute — shuffled partition same np
- redistribute
- redistribute different-np
- SerializerJSON vector round-trip
- SerializerJSON uint8 array round-trip
- SerializerJSON path operations
- SerializerJSON shared pointer deduplication
- SerializerH5 scalar round-trip
- SerializerH5 vector round-trip
- SerializerH5 distributed vector
- SerializerH5 uint8 distributed round-trip
- SerializerH5 path operations
- SerializerH5 string round-trip

## `dnds_test_source_organization`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_source_organization` |

[test_SourceOrganization.cpp](../../test/cpp/DNDS/test_SourceOrganization.cpp)

- Legacy canonical and forwarding headers preserve the same array type

## `dnds_test_source_organization_native`

| Level | CTest registrations |
|---|---|
| 1 | `dnds_source_organization_native` |

[test_SourceOrganizationNative.cpp](../../test/cpp/DNDS/test_SourceOrganizationNative.cpp)

- Canonical native access requirements are independent of legacy mesh arrays

## `euler_test_chem_ode`

| Level | CTest registrations |
|---|---|
| 1 | `euler_chem_ode` |

[test_ChemODE.cpp](../../test/cpp/Euler/test_ChemODE.cpp)

- Cantera custom const-volume affine species reactor
- Cantera custom affine reactor reduces target residual
- 0D const-vol — implicit Euler, species-only Newton, T via PhysicsProperties
- 0D const-vol — react_test history tracks Cantera reactor
- 0D const-vol helper uses physical Cantera scales
- Finite-difference Jacobian check at non-initial state
- FD Jacobian for pressure-dependent mechanism (GRI 3.0)

## `euler_test_evaluator_reactive`

| Level | CTest registrations |
|---|---|
| 1 | `euler_evaluator_reactive_np1`, `euler_evaluator_reactive_np2` |
| 2 | `euler_evaluator_reactive_np4`, `euler_evaluator_reactive_np8` |

[test_EulerEvaluatorReactive.cpp](../../test/cpp/Euler/test_EulerEvaluatorReactive.cpp)

- Physical source splitting rejects globally disabled sources
- EulerEvaluator repairs reactive cell-mean species

## `euler_test_limited_vr`

| Level | CTest registrations |
|---|---|
| 1 | `euler_limited_vr` |

[test_LimitedVR.cpp](../../test/cpp/Euler/test_LimitedVR.cpp)

- LVR selectors write names, accept retained integer aliases and reject removed choices
- Weak LVR branch preserves uniform-strip quadratics and rejects contacts and expansion
- Weak LVR branch detects rescaled shocks above its explicit noise floor
- Limited variational reconstruction cubic ramp has exact endpoints and bounded transition
- Weak LVR alpha relaxation preserves its fixed point and damps updates
- Limited variational reconstruction sensor rejects contacts shear and expansions
- Limited variational reconstruction sensor has a bounded intermediate response
- LVR expansion switch symmetrizes only the pressure and normal-velocity face gate
- Limited variational reconstruction compressive shock saturates and is orientation invariant
- Limited variational reconstruction density-only contact gate is separate and orientation invariant
- Limited variational reconstruction contact gate can be disabled exactly
- Limited variational reconstruction combines shock and contact gates by maximum
- Limited variational reconstruction settings parse and validate
- LVR input presets expand before explicit overrides and preserve legacy input

## `euler_test_rans_model`

| Level | CTest registrations |
|---|---|
| 1 | `euler_rans_model` |

[test_RANSModel.cpp](../../test/cpp/Physics/test_RANSModel.cpp)

- Shared SA property and diffusion extraction preserves the Euler formulas
- RANS adapter dispatch retains source and destruction diagonal conventions
- RANS transport source and diffusion are independent of Cartesian frame orientation
- SST-2003 source blends inner and outer production coefficients
- Status-returning RANS point math preserves the CPU closure and reports missing source geometry

## `euler_test_reactive_split_indicator`

| Level | CTest registrations |
|---|---|
| 1 | `euler_reactive_split_indicator` |

[test_ReactiveSplitIndicator.cpp](../../test/cpp/Euler/test_ReactiveSplitIndicator.cpp)

- Reactive split strategy names round trip and reject invalid configuration
- Chemical response ratio is step independent and has no auxiliary gates
- Local matched-rate ratio guards activity and forbids spatial expansion
- Balance chain filters preserve constants and defer endpoint snapping
- Transport balance has matched-rate limits and independent configuration
- Reactive split indicator endpoints and bounds
- Reactive split coupled score is shock suppressed
- Reactive split activity saturation exposes a0 b0 and p
- Reactive split activity tuning is parsed and validated
- Reactive split v2 rate ratio is scale invariant and configurable
- Hill switch is the default with an exact zero tail and configurable midpoint
- Chemistry escape preserves baseline and selects stiff low diffusion chemistry
- Neighbor expansion is bounded, decaying, and shock gated

## `euler_test_shared_physics`

| Level | CTest registrations |
|---|---|
| 1 | `euler_shared_physics` |

[test_Properties.cpp](../../test/cpp/Physics/test_Properties.cpp)

- Loaded model description detects changed kinetics with unchanged species
- shared mixture: fused SI outputs, scaled UV and legacy Euler properties agree
- shared mixture workers: independent concurrent contexts and retained lifetime
- shared mixture: workers reject a different loaded material even with identical filenames

## `euler_test_uv`

| Level | CTest registrations |
|---|---|
| 1 | `euler_uv` |

[test_UV.cpp](../../test/cpp/Euler/test_UV.cpp)

- setState_UV internal energy consistency
- setState_UV with evolved composition from ODE step 5

## `geom_test_boundary_interface`

| Level | CTest registrations |
|---|---|
| 1 | `geom_boundary_interface_np1`, `geom_boundary_interface_np2` |
| 2 | `geom_boundary_interface_np4`, `geom_boundary_interface_np8` |

[test_BoundaryInterface.cpp](../../test/cpp/Geom/test_BoundaryInterface.cpp)

- Boundary interface projection preserves query IDs and deterministic seams through rigid motion
- Interface search rejects uncovered points invalid controls duplicate IDs and unsupported distributed use
- Triangle and quadrilateral surface matching retains adjacent volume slots
- Boundary stencil preparation retains DSL chain endpoints and canonical ghost mappings
- Relative sliding changes donor cells while preserving the previous publication
- Coincident disconnected surface interiors are rejected rather than silently selecting a donor
- Rigid rotation through a full turn preserves matching and changes the physical donor side

## `geom_test_bounding_box_tree`

| Level | CTest registrations |
|---|---|
| 1 | `geom_bounding_box_tree` |

[test_BoundingBoxTree.cpp](../../test/cpp/Geom/test_BoundingBoxTree.cpp)

- Switchable BVH candidates equal exhaustive box scans including thin repeated and tiny boxes
- BVH wrappers own inputs share immutable copies and support concurrent const queries
- BVH rejects invalid boxes backends and query inputs before traversal

## `geom_test_entity_work_partition`

| Level | CTest registrations |
|---|---|
| 1 | `geom_entity_work_partition_np1`, `geom_entity_work_partition_np2` |
| 2 | `geom_entity_work_partition_np4`, `geom_entity_work_partition_np8` |

[test_EntityWorkPartition.cpp](../../test/cpp/Geom/test_EntityWorkPartition.cpp)

- Node-star work sets match an independent Cartesian ownership oracle
- Work classification rejects missing deeper closure without fetching rows

## `geom_test_mesh_audit`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_audit_np1`, `geom_mesh_audit_np2` |
| 2 | `geom_mesh_audit_np4`, `geom_mesh_audit_np8` |

[test_MeshAudit.cpp](../../test/cpp/Geom/test_MeshAudit.cpp)

- Audit batch 2: wall quadrature preserves square distances
- Audit batch 2: native smoothers preserve zero displacement and coordinates

## `geom_test_mesh_connectivity`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_connectivity_np1`, `geom_mesh_connectivity_np2` |
| 2 | `geom_mesh_connectivity_np4`, `geom_mesh_connectivity_np8` |

[test_MeshConnectivity.cpp](../../test/cpp/Geom/test_MeshConnectivity.cpp)

- Inverse: 4-quad serial correctness
- Inverse: round-trip covers original cell2node
- ComposeFiltered: 4-quad cell2cell via node-neighbor
- ComposeFiltered: face-share filter (minShared=dim)
- Regression: Inverse matches independent lattice node stars on UniformSquare_10
- Regression: ComposeFiltered matches independent lattice neighbors on UniformSquare_10
- Native registry cone management rejects duplicate and omitted definitions
- Native registry support management preserves semantic inverse identity
- Native cone typed access validates family state and ordered companions
- Native registry lookup retains the exact typed row lease after registry destruction
- Native support has explicit sorted entity semantics independent of node image companions
- Inverse: fixed-width face2cell input keeps exact sentinel semantics
- Periodic 2x2x2: node-count composition cannot replace physical face neighbors
- narrowAdjToFixed: variable-to-fixed-2 conversion

## `geom_test_mesh_connectivity_interpolate`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_connectivity_interpolate_np1`, `geom_mesh_connectivity_interpolate_np2` |
| 2 | `geom_mesh_connectivity_interpolate_np4`, `geom_mesh_connectivity_interpolate_np8` |

[test_MeshConnectivity_Interpolate.cpp](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp)

- Interpolate: 4-quad faces (2D)
- Interpolate: 2-tri faces (2D)
- Interpolate: single tet faces (3D)
- Interpolate: two tets sharing a face (3D)
- Interpolate: single hex faces (3D)
- Interpolate: single tet edges (3D)
- Interpolate: two-tet shared edges (3D)
- Interpolate: mixed tri+quad faces (2D)
- InterpolateGlobal: 4x4x4 distributed hex faces (non-periodic)
- InterpolateGlobal: 4x4x4 distributed hex faces (X-periodic)
- InterpolateGlobal: 4x4x4 distributed hex edges (non-periodic)
- InterpolateGlobal: 4x4x4 distributed hex edges (X-periodic)
- InterpolateGlobal: 4x4x4 distributed hex faces (triply-periodic)
- Regression: Interpolate matches InterpolateFace on UniformSquare_10
- Interpolate: 2x2 periodic quad mesh — collaborating check required
- Periodic 2x2x2: face interpolation (3D, collaborating check)
- Periodic 2x2x2: edge interpolation (3D, collaborating check)

## `geom_test_mesh_current_output`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_current_output_np1`, `geom_mesh_current_output_np2` |
| 2 | `geom_mesh_current_output_np4`, `geom_mesh_current_output_np8` |

[test_MeshCurrentOutput.cpp](../../test/cpp/Geom/test_MeshCurrentOutput.cpp)

- Current periodic VTKHDF maps point callbacks from detached numbering to source rows
- Prepared VTKHDF worker keeps reductions on its output communicator
- Native VTKHDF owns output buffers without legacy mesh conversion
- Native serial and distributed VTU and Tecplot preserve rows and periodic images
- Native output rejects cross-rank publication and periodic schedule mismatches
- CGNS output rejects a globally empty native mesh before file creation
- Native CGNS output round trips independently owned nodes and empty ranks
- Native primary compatibility archive streams payloads including boundary curves
- Output rejects unpublished authority before file creation and permits native retry

## `geom_test_mesh_distributed_read`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_distributed_read_np1`, `geom_mesh_distributed_read_np2` |
| 2 | `geom_mesh_distributed_read_np4`, `geom_mesh_distributed_read_np8` |

[test_MeshDistributedRead.cpp](../../test/cpp/Geom/test_MeshDistributedRead.cpp)

- SameNp: global cell count
- SameNp: global node count
- SameNp: global bnd count
- SameNp: global face count
- SameNp: expected counts from config
- SameNp: every rank has cells
- SameNp: face2cell valid
- SameNp: node2cell non-empty
- SameNp: cell2cellOrig globally unique
- SameNp: coordinate bounding box is sane
- CrossNp: global cell count
- CrossNp: global node count
- CrossNp: global bnd count
- CrossNp: global face count
- CrossNp: every rank has cells
- CrossNp: face2cell valid
- CrossNp: cell2cellOrig globally unique
- CrossNp: node2cell non-empty
- CrossNp: helper path PrepareMesh resolves owned node2cell

## `geom_test_mesh_edit`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_edit_np1`, `geom_mesh_edit_np2` |
| 2 | `geom_mesh_edit_np4`, `geom_mesh_edit_np8` |

[test_MeshEdit.cpp](../../test/cpp/Geom/test_MeshEdit.cpp)

- Native geometry edits preserve detached snapshots and escaped value copies
- Staged metadata and neighbor-order edits validate before publication
- Uncommitted scopes discard copies and active reads block edit publication
- Mesh edits require explicit native authority and recover after collective rejection
- Native-only edits preserve namespace IDs named definitions and immutable source buffers

## `geom_test_mesh_entity_field`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_entity_field_np1`, `geom_mesh_entity_field_np2` |
| 2 | `geom_mesh_entity_field_np4`, `geom_mesh_entity_field_np8` |

[test_MeshEntityField.cpp](../../test/cpp/Geom/test_MeshEntityField.cpp)

- Entity-bound fields reject wrong domain/kind and retain field/mesh leases
- Detached publication binds active and inactive fields to final native destination and rejects old routes
- Entity field collective setup rejects rank-local stale/shape/schema and transfers explicit POD layouts
- Entity fields migrate table padded and CSR storage independently of view policy
- Entity CSR fields preserve variable row shape values and source leases across empty owners

## `geom_test_mesh_entity_field_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_entity_field_cuda_np2` |

[test_MeshEntityField_CUDA.cu](../../test/cpp/Geom/test_MeshEntityField_CUDA.cu)

- Native EntityField CUDA scopes and direct multi-GPU halo retain checked association

## `geom_test_mesh_facade`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_facade_np1`, `geom_mesh_facade_np2` |
| 2 | `geom_mesh_facade_np4`, `geom_mesh_facade_np8` |

[test_MeshFacade.cpp](../../test/cpp/Geom/test_MeshFacade.cpp)

- Physical cell pairs publish fixed rows while sharing CSR and padded payloads
- Legacy read facade matches independent legacy arrays without copying native rows
- Legacy facade distinguishes unavailable groups from published empty groups
- Legacy facade preserves owned and ghost segments with outer closure sentinels
- Legacy read owner retains native storage through moves and mesh handle replacement
- Mesh facade host acquisition exposes exact local rows and semantic helper behavior
- Mesh facade scopes retain storage after source deletion and reject invalid contexts
- Native scope and detached capture expose exactly the requested optional groups
- Native facade construction rejects mismatched shapes and owned closure before publication
- Native snapshot validates reciprocal owned faces and partial outer ghost closure
- Native publication retains aliased storage and releases it with the final scope
- Mesh host view getters reject moved and sealed scopes
- Native entity spaces export isolated field maps and match their native halo route

## `geom_test_mesh_facade_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_facade_cuda` |

[MeshViewKernel.cu](../../test/cpp/Geom/MeshViewKernel.cu)


[test_MeshFacade.cpp](../../test/cpp/Geom/test_MeshFacade.cpp)

- Physical cell pairs publish fixed rows while sharing CSR and padded payloads
- Legacy read facade matches independent legacy arrays without copying native rows
- Legacy facade distinguishes unavailable groups from published empty groups
- Legacy facade preserves owned and ghost segments with outer closure sentinels
- Legacy read owner retains native storage through moves and mesh handle replacement
- Mesh facade host acquisition exposes exact local rows and semantic helper behavior
- Mesh facade scopes retain storage after source deletion and reject invalid contexts
- Native scope and detached capture expose exactly the requested optional groups
- Native facade construction rejects mismatched shapes and owned closure before publication
- Native snapshot validates reciprocal owned faces and partial outer ghost closure
- Native publication retains aliased storage and releases it with the final scope
- Mesh host view getters reject moved and sealed scopes
- Native entity spaces export isolated field maps and match their native halo route

[test_MeshFacade_CUDA.cu](../../test/cpp/Geom/test_MeshFacade_CUDA.cu)

- CUDA physical pair publication normalizes CSR and distinct padded owned ghost strides
- Mesh facade CUDA kernels traverse retained snapshots with host parity
- Moved mesh CUDA access can outlive its destroyed acquisition stream
- Mesh facade CUDA completion validates the acquisition stream
- Native mesh CUDA queries match existing const legacy pair views
- Detached native CUDA mesh backing is placed once and reused by reads
- Native builder CUDA stencil retains independent arrays through pending completion

## `geom_test_mesh_facade_overlap`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_facade_overlap_np1`, `geom_mesh_facade_overlap_np2` |
| 2 | `geom_mesh_facade_overlap_np4`, `geom_mesh_facade_overlap_np8` |

[test_MeshFacadeOverlap.cpp](../../test/cpp/Geom/test_MeshFacadeOverlap.cpp)

- Real mesh finite-volume neighbor sums overlap native halo with inline and worker CPU execution

## `geom_test_mesh_facade_overlap_cuda`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_facade_overlap_cuda_np2` |

[MeshFacadeOverlapKernel.cu](../../test/cpp/Geom/MeshFacadeOverlapKernel.cu)


[test_MeshFacadeOverlap_CUDA.cu](../../test/cpp/Geom/test_MeshFacadeOverlap_CUDA.cu)

- Real mesh CUDA finite-volume core uses owned solution during direct multi-GPU MPI halo

## `geom_test_mesh_facade_pipeline`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_facade_pipeline_np1`, `geom_mesh_facade_pipeline_np2` |
| 2 | `geom_mesh_facade_pipeline_np4`, `geom_mesh_facade_pipeline_np8` |

[test_MeshFacadePipeline.cpp](../../test/cpp/Geom/test_MeshFacadePipeline.cpp)

- Mesh facade checkpoint restores exact serial geometry and native query backing
- Mesh facade real CGNS pipeline preserves serial geometry through transitions

## `geom_test_mesh_field_checkpoint`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_field_checkpoint_np1`, `geom_mesh_field_checkpoint_np2` |
| 2 | `geom_mesh_field_checkpoint_np4`, `geom_mesh_field_checkpoint_np8` |

[test_MeshFieldCheckpoint.cpp](../../test/cpp/Geom/test_MeshFieldCheckpoint.cpp)

- Retained mesh and multi-history field form a coherent matching-partition H5 restart

## `geom_test_mesh_geometry_edit`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_geometry_edit_np1`, `geom_mesh_geometry_edit_np2` |
| 2 | `geom_mesh_geometry_edit_np4`, `geom_mesh_geometry_edit_np8` |

[test_MeshGeometryEdit.cpp](../../test/cpp/Geom/test_MeshGeometryEdit.cpp)

- Rigid geometry edit transforms coordinates periodic frame and wall vectors coherently
- Boundary rectification reduces remote node masks before refreshing coordinate ghosts
- Pure native geometry transforms preserve legacy scalar and rotation bit patterns
- Native geometry transforms preserve order identity topology and detached readers
- Native rectification reduces masks across independent owners and rejects callback failure
- Collective geometry operations reject a rank without native authority before staging
- Native arbitrary coordinate callbacks run on owners and refresh ghosts transactionally
- Native affine geometry maps periodic physical corners and invalidates wall vectors atomically
- Native affine geometry rejects nonrigid periodic conjugation before publication
- Near-plane native snapping edits canonical periodic owners and preserves physical images
- Geometry operations reject an unpublished mesh without changing its epoch

## `geom_test_mesh_mutation`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_mutation_np1`, `geom_mesh_mutation_np2` |
| 2 | `geom_mesh_mutation_np4`, `geom_mesh_mutation_np8` |

[test_MeshMutation.cpp](../../test/cpp/Geom/test_MeshMutation.cpp)

- Managed public mutations reject retained readers while detached conversion remains pure
- Native topology publication changes each declared generation once and auxiliary setup remains pure
- Failed nested mutation stays failed even when an intermediate caller catches it
- Public callback reentry cannot adopt a writer and staged failures preserve the source
- Bisection creates independent geometry under independent mesh mutation controllers
- Native primary serialization shares immutable shape safely with readers and preserves the source
- Collective native topology setup rejects a one-rank reader while output remains readonly
- Boundary destination rejection preserves both native authorities and controllers

## `geom_test_mesh_reorder`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_reorder_np1`, `geom_mesh_reorder_np2` |
| 2 | `geom_mesh_reorder_np4`, `geom_mesh_reorder_np8` |

[test_MeshReorder.cpp](../../test/cpp/Geom/test_MeshReorder.cpp)

- classifyAdj basic classification
- ReorderRegistry register and query
- ReorderPlan::apply cell-only local permutation
- ReorderPlan::apply node-only remap
- PermutationTransfer + buildLookup: reverse permutation value tracking
- PermutationTransfer distributed value tracking cross-rank
- PermutationTransfer::buildLookup cross-rank resolve with pullSet
- Native field uses exact local permutation and preserves old aliases
- Explicit transfer layout preserves static stride and rejects one-rank capacity errors before publication
- Staged field bundle preserves every source when a later preparation fails
- Staged Array registration rejects duplicate destinations and permits distinct old aliases
- ReorderPlan::apply RELOCATE_REMAP (both source and target reordered)
- NativeRowTransfer exact native routes with empty ranks and skew
- Audit regression: local cell reorder retains periodic face rows
- ReorderEntities cell-only local on UniformSquare_10
- ReorderEntities cell-only with face destruction on UniformSquare_10
- ReorderEntities cell distributed round-robin with follow
- ReorderEntities node-only local on UniformSquare_10
- ReorderEntities with external companion array (solver-like)
- Staged field attachment rejection precedes mesh mutation
- buildReorderRegistry populates pullSets for off-rank references
- ReorderEntities: edge destroy reconstructs all edge arrays and state
- ReorderEntities: combined face and edge destroy reconstructs both groups
- ReorderEntities: face destroy reconstructs cell2facePbi on periodic mesh
- ReorderEntities: face-preserving cell reorder reattaches face arrays
- buildReorderRegistry pull-sets include edge sources after interpolation
- buildReorderRegistry rejects non-global face adj after face interpolation
- buildReorderRegistry registers cell2facePbi as cell companion for periodic mesh
- ReorderEntities: node companion arrays reattached after reorder
- ReorderEntities: bnd2faceV and face2bndM reconstructed on Face destroy
- InterpolateEdge: edge arrays built and adjEdgeState is Adj_PointToLocal
- InterpolateEdge: AdjLocal2GlobalEdge / AdjGlobal2LocalEdge round-trip
- ReorderLocalCells: existing edges retain exact IDs and rows
- ReorderEntities: edge-preserving cell reorder reattaches edge arrays
- ReorderEntities: Node-explicit distributed round-robin
- ReorderEntities: Bnd-explicit distributed round-robin
- ReorderEntities: FollowSpec Bnd follows Node via bnd2node
- ReorderEntities: face-neighbor relation has independent target lookup coverage
- ReorderLocalCells preserves multi-layer ghost identities and connectivity
- ReorderEntities: Face-explicit identity and arriving cell rows
- ReorderEntities: Cell explicit, all topology follows Cell on 3D mesh
- ReorderEntities: Node explicit, all topology follows Node on 3D mesh

## `geom_test_mesh_transition`

| Level | CTest registrations |
|---|---|
| 1 | `geom_mesh_transition_np1`, `geom_mesh_transition_np2` |
| 2 | `geom_mesh_transition_np4`, `geom_mesh_transition_np8` |

[test_MeshTransition.cpp](../../test/cpp/Geom/test_MeshTransition.cpp)

- Alias candidates require one common earlier payload across ranks
- Prepared transition uses a stable destination slot across moves and rejects expiry
- Detached transition retains source scopes and publishes matching native fields
- Detached periodic transition retains geometry and relation companion rows
- Detached preparation failures retain mesh and every field source
- Detached transition rejects unsupported inventory before preparing candidates
- Inactive registered fields still require exact owned source rows
- Native transition invalidates destination extraction row caches without editing source
- Delayed publication rejects rank-local native field edits and replacement
- Native output preparation and primary checkpoint preserve authoritative rows and wall policy
- Detached transition reconstructs 3D edges under an explicit cache policy
- Detached transition preserves independently placed edges with exact native fields and companion slots

## `geom_test_native_mesh_archive`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_archive_np1`, `geom_native_mesh_archive_np2` |
| 2 | `geom_native_mesh_archive_np4`, `geom_native_mesh_archive_np8` |

[test_NativeMeshArchive.cpp](../../test/cpp/Geom/test_NativeMeshArchive.cpp)

- Native archive preserves a full custom ghost recipe across saved and even partitions
- Native H5 archive preserves exact topology identity and stencil across saved and even partitions
- Native per-rank JSON archive preserves matching topology and rejects incompatible collective read modes
- Native archive preserves all five entity families periodic frames and wall vectors without legacy storage

## `geom_test_native_mesh_assertions`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_assertions_np1`, `geom_native_mesh_assertions_np2` |
| 2 | `geom_native_mesh_assertions_np4`, `geom_native_mesh_assertions_np8` |

[test_NativeMeshAssertions.cpp](../../test/cpp/Geom/test_NativeMeshAssertions.cpp)

- Native mesh assertions validate real incidences and reject missing groups and reciprocal defects
- Native face assertions permit unavailable outer ghost cell faces and preserve self periodic multiplicity

## `geom_test_native_mesh_builder`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_builder_np1`, `geom_native_mesh_builder_np2` |
| 2 | `geom_native_mesh_builder_np4`, `geom_native_mesh_builder_np8` |

[test_NativeMeshBuilder.cpp](../../test/cpp/Geom/test_NativeMeshBuilder.cpp)

- Owned primary redistribution and face construction finalize once with the supplied boundary ring
- Native primary builder closes remote owned-node stars explicitly
- Native primary builder matches all boundary nodes and transfers independent companions
- Native primary builder retains every cell layer and its node closure
- Native primary builder normalizes uniform CSR and padded companions without copying values
- Native primary builder rejects a boundary diagonal collectively then accepts a corrected face
- Native primary builder includes empty ranks and agrees on invalid input

## `geom_test_native_mesh_caches`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_caches_np1`, `geom_native_mesh_caches_np2` |
| 2 | `geom_native_mesh_caches_np4`, `geom_native_mesh_caches_np8` |

[test_NativeMeshCaches.cpp](../../test/cpp/Geom/test_NativeMeshCaches.cpp)

- Coordinate compatibility hash matches legacy owned and ghost rows before and after redistribution
- Native face graph and direct ordering match exact face slots and generic serial algorithms
- Native direct setup keeps empty cell owners in RCM collectives

## `geom_test_native_mesh_cgns`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_cgns_np1`, `geom_native_mesh_cgns_np2` |
| 2 | `geom_native_mesh_cgns_np4`, `geom_native_mesh_cgns_np8` |

[test_NativeMeshCGNS.cpp](../../test/cpp/Geom/test_NativeMeshCGNS.cpp)

- Native CGNS serial rows and distributed construction preserve exact topology
- Native CGNS import errors agree collectively and permit valid retry

## `geom_test_native_mesh_construction`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_construction_np1`, `geom_native_mesh_construction_np2` |
| 2 | `geom_native_mesh_construction_np4`, `geom_native_mesh_construction_np8` |

[test_NativeMeshConstruction.cpp](../../test/cpp/Geom/test_NativeMeshConstruction.cpp)

- Native arrays publish a mesh with empty legacy storage and retained read-only aliases
- Native publication rejects missing topology, invalid rows and aliased active writers
- Native publication coordinates rank-local missing spaces and topology
- Native publication rejects persistent cell IDs owned on multiple ranks
- Native ghost IDs agree with their owner and failed checks retain source spaces
- Native mesh namespaces and non-reuse high-water marks survive publication
- Native publication rejects inconsistent identity and reused or exhausted IDs
- Native publication rejects optional values outside their validated groups
- Native element dimensions follow cell and boundary roles
- Requested native wall distance is validated exposed and protected
- Native mesh and entity descriptors retain their managed communicator
- Native entity spaces support remote node-star discovery without a legacy mesh
- Replacing an entity descriptor does not mutate retained source controllers
- Native memory accounting counts retained capacities and shared named identity ancestry arrays once
- Publication audit policy defaults full and explicitly trusted IDs remain re-auditable
- Publication identity diagnostics detect cross-rank duplicate and owner-inconsistent ghost IDs
- Trusted closure skips reciprocity diagnostics while preserving mandatory shape range and sentinels
- Audit policy flags agree collectively during construction and checkpoint audit
- Effective construction audit policy propagates through native setup and can be overridden for derivation
- Checkpoint audits collectively reject mixed publication inventory or namespace before identity routes

## `geom_test_native_mesh_extraction`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_extraction_np1`, `geom_native_mesh_extraction_np2` |
| 2 | `geom_native_mesh_extraction_np4`, `geom_native_mesh_extraction_np8` |

[test_NativeMeshExtraction.cpp](../../test/cpp/Geom/test_NativeMeshExtraction.cpp)

- Native boundary extraction compacts owner nodes and preserves ordered high-order geometry
- Native boundary extraction handles an empty selection without retaining unused nodes
- Native boundary extraction retains periodic tangential images and rejects inconsistent flags
- Native boundary facade retains source reads and publishes legacy parent caches explicitly
- Native boundary parent cache marks newly introduced ghosts absent from source closure
- Boundary facade rejects absent native authority collectively before mutating either mesh

## `geom_test_native_mesh_legacy_archive`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_legacy_archive_np1`, `geom_native_mesh_legacy_archive_np2` |
| 2 | `geom_native_mesh_legacy_archive_np4`, `geom_native_mesh_legacy_archive_np8` |

[test_NativeMeshLegacyArchive.cpp](../../test/cpp/Geom/test_NativeMeshLegacyArchive.cpp)

- Legacy primary JSON and H5 decode directly to native owned rows
- Legacy single-rank H5 imports with changed MPI size through native even split
- Legacy periodic primary import preserves exact ordered image bits and original IDs
- Native archive helper coordinates rank-local missing and malformed JSON before retry
- Legacy periodic high-order import rejects explicitly duplicated node IDs without renumbering

## `geom_test_native_mesh_local_reorder`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_local_reorder_np1`, `geom_native_mesh_local_reorder_np2` |
| 2 | `geom_native_mesh_local_reorder_np4`, `geom_native_mesh_local_reorder_np8` |

[test_NativeMeshLocalReorder.cpp](../../test/cpp/Geom/test_NativeMeshLocalReorder.cpp)

- Native local reorder shares permutation semantics and preserves current identity payloads
- Native facade local reorder validates inputs and publishes only after readers release
- Primary local reorder matches full facial graph permutation and final partition graph
- Overlap order groups all node-star core cells before dependent workers
- Periodic primary local reorder uses frame-aware facial adjacency
- Local cell reorder accepts padded combined IDs and retains noncell payload after source release
- Shared local partition helper handles disconnected empty and singleton face graphs

## `geom_test_native_mesh_openfoam`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_openfoam_np1`, `geom_native_mesh_openfoam_np2` |
| 2 | `geom_native_mesh_openfoam_np4`, `geom_native_mesh_openfoam_np8` |

[test_NativeMeshOpenFOAM.cpp](../../test/cpp/Geom/test_NativeMeshOpenFOAM.cpp)

- Native OpenFOAM canonical Hex8 matches fixed slots and boundary remap
- Native serial reader-owned policy and already-O2 derivation retain intended order
- Source setup partitions parents before refinement and retains parent-owned children

## `geom_test_native_mesh_partition_graph`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_partition_graph_np1`, `geom_native_mesh_partition_graph_np2` |
| 2 | `geom_native_mesh_partition_graph_np4`, `geom_native_mesh_partition_graph_np8` |

[test_NativeMeshPartitionGraph.cpp](../../test/cpp/Geom/test_NativeMeshPartitionGraph.cpp)

- Reader primary unweighted graph equals full facial publication and partitions identical primary rows
- Reader scratch rejects dimension-one boundary meshes collectively
- Face deferral agrees collectively and preserves weighted fallback
- Reader graph retains empty disconnected nonmanifold and cracked high-order behavior
- Native partition uses exact global CSR and normalized nonuniform physical face diameters
- Several physical faces of one cell pair contribute their maximum diameter

## `geom_test_native_mesh_periodic`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_periodic_np1`, `geom_native_mesh_periodic_np2` |
| 2 | `geom_native_mesh_periodic_np4`, `geom_native_mesh_periodic_np8` |

[test_NativeMeshPeriodic.cpp](../../test/cpp/Geom/test_NativeMeshPeriodic.cpp)

- Native periodic strip constructs exact self incidences and preserves them through archive and redistribution
- Native periodic primary rejects invalid image metadata collectively and retains valid input
- Exact periodic frame matching handles repeated IDs whose XOR changes sorted order
- Periodic frame matching agrees with independent full image search
- One-cell doubly periodic native torus retains two distinct self faces
- Partitioned native periodic cells distinguish internal physical and periodic faces
- Quadratic native periodic topology keeps image-distinct spans and rejects cracked midside nodes
- Native periodic construction retains a rigid rotation in physical cell images
- Native periodic elevation bisection and external extraction preserve physical geometry and ancestry
- One-cell periodic elevation separates tangent spans with repeated canonical IDs
- Three-dimensional periodic refinement preserves exact volume persistent images and ancestry across redistribution and restart
- Periodic parent-face flags use the published owner frame across ranks

## `geom_test_native_mesh_preparation`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_preparation_np1`, `geom_native_mesh_preparation_np2` |
| 2 | `geom_native_mesh_preparation_np4`, `geom_native_mesh_preparation_np8` |

[test_NativeMeshPreparation.cpp](../../test/cpp/Geom/test_NativeMeshPreparation.cpp)

- Owned graph rejects mismatched weight branches before coordinate exchange
- Owned CGNS partition permutation and faces need only one final assembly
- Periodic default boundary ownership follows physical side while explicit support follows both
- Source helpers retain exact legacy serial METIS graph policy and cell ownership
- Prepared reordered faces follow independent post-permutation parent-major slots
- Native facial partition retains ID-derived fields geometry and sparse closure
- Native primary partition adds facial IDs while retaining exact original field association
- Native partition invalid options reject collectively without mutating source
- Native production preparation preserves facade identity rejects readers and writes serial output
- Partition preserves independent existing Face ownership and exact source associations for registered fields

## `geom_test_native_mesh_publication`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_publication_np1`, `geom_native_mesh_publication_np2` |
| 2 | `geom_native_mesh_publication_np4`, `geom_native_mesh_publication_np8` |

[test_NativeMeshPublication.cpp](../../test/cpp/Geom/test_NativeMeshPublication.cpp)

- Managed mesh readers retain state metadata across placement replacement and moves
- Native publication owns semantic queries and current numbering without a legacy workspace
- Published entity maps cannot be changed through a solver field export
- Published periodic rows retain readonly incidence companions
- Native managed replacement rejects reads and preserves detached snapshots
- Native original IDs reject absent authority invalid rows and malformed candidate extents
- Native symbolic factorization retains detached setup metadata exactly

## `geom_test_native_mesh_refinement`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_refinement_np1`, `geom_native_mesh_refinement_np2` |
| 2 | `geom_native_mesh_refinement_np4`, `geom_native_mesh_refinement_np8` |

[test_NativeMeshRefinement.cpp](../../test/cpp/Geom/test_NativeMeshRefinement.cpp)

- Owned elevation retains legacy original labels independently of current global row numbering
- Native O2 bisection preserves independent exact child cones volume and durable parent IDs
- Native bisection retains boundary labels ancestry and empty-rank participation
- Native bisection rejects first-order input collectively without changing source
- Bisection sparse persistent ancestry survives source destruction redistribution and native restart
- Native bisection ancestry rejects malformed rank-local publication input
- CSR zero and multiple source ancestry retains aliases and roundtrips redistribution with empty ranks
- Native elevation deduplicates shared quadratic spans and preserves exact slots across MPI owners

## `geom_test_native_mesh_scenarios`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_scenarios_np1`, `geom_native_mesh_scenarios_np2` |
| 2 | `geom_native_mesh_scenarios_np4`, `geom_native_mesh_scenarios_np8` |

[test_NativeMeshScenarios.cpp](../../test/cpp/Geom/test_NativeMeshScenarios.cpp)

- High valence all-remote owned node has exact closure and SUM gather scatter
- Node cell node and node edge node have independent Hex8 bridge closure
- Physical periodic face variants traverse direct any all exact ordered definitions
- Unrelated remote components do not grow rank0 traversal C++ scratch

## `geom_test_native_mesh_serial_output`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_serial_output_np1`, `geom_native_mesh_serial_output_np2` |
| 2 | `geom_native_mesh_serial_output_np4`, `geom_native_mesh_serial_output_np8` |

[test_NativeMeshSerialOutput.cpp](../../test/cpp/Geom/test_NativeMeshSerialOutput.cpp)

- Native serial output gathers exact global rows to a selectable reader with maps-only field compatibility
- Native serial output retains old geometry and explicitly refreshes after publication
- Serial output facade rejects absent native authority and retains successful descriptors after failure

## `geom_test_native_mesh_setup`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_setup_np1`, `geom_native_mesh_setup_np2` |
| 2 | `geom_native_mesh_setup_np4`, `geom_native_mesh_setup_np8` |

[test_NativeMeshSetup.cpp](../../test/cpp/Geom/test_NativeMeshSetup.cpp)

- Native setup replacement stages identity and invalidates prior associations
- Native setup rejects null malformed association and preserves an existing publication on failure
- Archive helper repartition retains requested second ghost ring
- Native helper guards reject before publishing or creating archive files

## `geom_test_native_mesh_smoothing`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_smoothing_np1`, `geom_native_mesh_smoothing_np2` |
| 2 | `geom_native_mesh_smoothing_np4`, `geom_native_mesh_smoothing_np8` |

[test_NativeMeshSmoothing.cpp](../../test/cpp/Geom/test_NativeMeshSmoothing.cpp)

- Native smoothing Hermite boundary oracle and detached local RBF
- Native smoothing no-op and collective callback/options failure retry
- Native global RBF and elasticity preserve exact boundary constraints
- Native 3D smoothing preserves independent octahedral Hermite geometry after redistribution
- Native periodic flat smoothing retains exact physical images in all three solvers
- Native facade smoothing stages work, rejects readers and stale caches, then retries atomically
- Retired compact support smoother rejects explicitly without mutating native source

## `geom_test_native_mesh_snapshot`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_snapshot_np1`, `geom_native_mesh_snapshot_np2` |
| 2 | `geom_native_mesh_snapshot_np4`, `geom_native_mesh_snapshot_np8` |

[test_NativeMeshSnapshot.cpp](../../test/cpp/Geom/test_NativeMeshSnapshot.cpp)

- Canonical incidence projection shares host rows and guards source numbering
- Detached native mesh snapshots retain values metadata and independent validity
- Detached snapshot publication retains validation and explicit placement

## `geom_test_native_mesh_topology`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_mesh_topology_np1`, `geom_native_mesh_topology_np2` |
| 2 | `geom_native_mesh_topology_np4`, `geom_native_mesh_topology_np8` |

[test_NativeMeshTopology.cpp](../../test/cpp/Geom/test_NativeMeshTopology.cpp)

- Native face completion matches independent topology with remote boundary ownership and empty ranks
- Native tetrahedron face and edge completion preserves exact incident cell sets
- Native topology rejects nonmanifold faces and duplicate boundary assignments collectively
- Native topology repairs missing parent ghosts from authoritative owned node incidences
- Construction closure tree agrees across complete graph and node-star inputs
- Three tetrahedra retain a shared edge with three incident parents
- Native topology rejects shared quadratic edge vertices with inconsistent midpoint identity
- Generated topology keeps persistent IDs and oriented slots through archive and independent redistribution
- Ordinary native topology refresh recovers every high ID after reorder and independent redistribution
- Native topology refresh matches exact periodic occurrence keys and preserves named ancestry
- Native refresh allocates only unmatched keys and rejects uninferable retained associations collectively
- Periodic three dimensional refresh preserves repeated-node face and edge keys
- Distinct full topology keys retain different IDs even under an exact rendezvous hash collision
- Intermediate-only named definitions reject added topology keys without inventing new paths
- Topology ID reuse distinguishes higher order vertex roles despite equal full node image multisets
- Quadratic topology roles reject actual cracked shared faces and prevent stale generated ID reuse
- Canonical topology role keys equal the independent full-image reference for every supported subentity

## `geom_test_native_relation_conversion`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_relation_conversion_np1`, `geom_native_relation_conversion_np2` |
| 2 | `geom_native_relation_conversion_np4`, `geom_native_relation_conversion_np8` |

[test_NativeRelationConversion.cpp](../../test/cpp/Geom/test_NativeRelationConversion.cpp)

- Native conversion preserves owned and ghost shape slots and unavailable negative encodings
- Local relation validation checks ghost bounds global ordering and ghost source self incidences
- Retained relation validation follows its read scope and fresh reads validate changed values
- Native conversion preserves dense stride and logical padded CSR rows on empty owners
- Native current numbering conversion survives local reorder redistribution reclosure and empty owners
- Periodic conversion keeps repeated canonical IDs ordered frames and distinct persistent IDs

## `geom_test_native_relation_registry`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_relation_registry_np1`, `geom_native_relation_registry_np2` |
| 2 | `geom_native_relation_registry_np4`, `geom_native_relation_registry_np8` |

[test_NativeRelationRegistry.cpp](../../test/cpp/Geom/test_NativeRelationRegistry.cpp)

- Generic adjacency ghost routes and named publications preserve storage including empty ghosts
- Empty table and padded adjacency publications retain distinct storage families
- Adjacency registry comparison preserves wire identity without serialization
- Named native relation variants retain exact definitions arrays and canonical orders
- Native mesh registration retains named inverse and selection variants beside facade adapters
- Mathematical inverse definitions never alias legacy physical slots
- Named native relation publication rejects invalid rank-local state and collective definition mismatch
- Duplicate facade definitions require identical decoded ordered slots
- Named relation access rejects stale publication epochs
- Sorted native relation order uses globals across owned and ghost local indices
- Native removeSelf descriptor rejects self incidences
- Adjacency scan cache follows alias writes and retains read authority

## `geom_test_native_transition_rebuild`

| Level | CTest registrations |
|---|---|
| 1 | `geom_native_transition_rebuild_np1`, `geom_native_transition_rebuild_np2` |
| 2 | `geom_native_transition_rebuild_np4`, `geom_native_transition_rebuild_np8` |

[test_NativeTransitionRebuild.cpp](../../test/cpp/Geom/test_NativeTransitionRebuild.cpp)

- Native-only repeated face/edge reconstruction reserves IDs and stages exact retained output
- Native transition reconstruction retains surviving derived ancestry through archive
- Face-only periodic reconstruction preserves retained Edge IDs and every ordered frame slot
- Generated native ID floors agree collectively and never modify retained family high-water marks

## `geom_test_relation_access`

| Level | CTest registrations |
|---|---|
| 1 | `geom_relation_access_np1`, `geom_relation_access_np2` |
| 2 | `geom_relation_access_np4`, `geom_relation_access_np8` |

[test_RelationAccess.cpp](../../test/cpp/Geom/test_RelationAccess.cpp)

- Storage-neutral inverse and compose preserve sorted incidence and empty rows
- Relation reader distinguishes absence unavailable globals and wrong entity orders
- Moved relation reader rejects access while destination retains values and metadata
- Native inverse routes incidence owners with empty source and target ranks
- Inverse collectively rejects bad global payload before native communication
- Pinned setup mapping retains bounds conversion and moved-scope errors
- Typed native halo route rejects stale entity orders

## `geom_test_surface_point_map`

| Level | CTest registrations |
|---|---|
| 1 | `geom_surface_point_map_np1`, `geom_surface_point_map_np2` |
| 2 | `geom_surface_point_map_np4`, `geom_surface_point_map_np8` |

[test_SurfacePointMap.cpp](../../test/cpp/Geom/test_SurfacePointMap.cpp)

- Element geometry projects analytically known line triangle and warped quad points
- Surface snapshots freeze shared storage and local maps retain their geometry
- Distributed mesh-independent surface location agrees across routing modes and transfers element data

## `solver_test_linear`

| Level | CTest registrations |
|---|---|
| 1 | `solver_linear` |

[test_Linear.cpp](../../test/cpp/Solver/test_Linear.cpp)

- GMRES: solve 3x3 SPD system exactly
- GMRES: solve 10x10 random SPD system
- GMRES: diagonal preconditioner improves convergence
- PCG: solve 5x5 SPD system

## `solver_test_ode`

| Level | CTest registrations |
|---|---|
| 1 | `solver_ode` |

[test_ODE.cpp](../../test/cpp/Solver/test_ODE.cpp)

- SSPRK3: 3rd-order on oscillator
- ImplicitEuler: 1st-order on oscillator
- SDIRK4 (sc=0): 4th-order on oscillator
- ESDIRK4 (sc=1): 4th-order on oscillator
- ESDIRK3 (sc=2): 3rd-order on oscillator
- Trapezoid (sc=3): 2nd-order on oscillator
- ESDIRK2 (sc=4): 2nd-order on oscillator
- VBDF k=1: 1st-order on oscillator
- VBDF k=2: 2nd-order on oscillator
- DITR U2R2 (mask=0, alpha=0.5): 4th-order on oscillator
- DITR U2R2 (mask=0, alpha=0.55): 3rd-order on oscillator
- DITR U2R1 (mask=1): 3rd-order L-stable on oscillator
- SSPRK3: golden value on oscillator

## `solver_test_scalar`

| Level | CTest registrations |
|---|---|
| 1 | `solver_scalar` |

[test_Scalar.cpp](../../test/cpp/Solver/test_Scalar.cpp)

- BisectSolveLower: x^2 = 4 => x = 2
- BisectSolveLower: few iterations gives lower accuracy

## `cauchy_test_interface_cuda`

| Level | CTest registrations |
|---|---|
| 2 | `cauchy_interface_cuda_np1`, `cauchy_interface_cuda_np2` |

[test_InterfaceCUDA.cu](../../test/cpp/Cauchy/test_InterfaceCUDA.cu)

- Resident interface refresh reuses routes in colocated and dislocated placement

## `cfv_test_distributed_interface_sampling_cuda`

| Level | CTest registrations |
|---|---|
| 2 | `cfv_distributed_interface_sampling_cuda_np1`, `cfv_distributed_interface_sampling_cuda_np2` |

[test_DistributedInterfaceSamplingCUDA.cu](../../test/cpp/CFV/test_DistributedInterfaceSamplingCUDA.cu)

- Queued owner-side interface sampling agrees on inline worker and CUDA

## `cfv_test_limited_vr_cuda`

| Level | CTest registrations |
|---|---|
| 2 | `cfv_limited_vr_cuda_np1`, `cfv_limited_vr_cuda_np2` |

[test_LimitedVRCUDA.cu](../../test/cpp/CFV/test_LimitedVRCUDA.cu)

- Queued LVR seeds and neighbor rounds retain CPU semantics across host workers and CUDA
- Limited CUDA Jacobi matches CPU relaxation and excludes neighbors for a fixed target

## `cfv_test_reconstruction`

| Level | CTest registrations |
|---|---|
| 2 | `cfv_reconstruction_np1`, `cfv_reconstruction_np2`, `cfv_reconstruction_np4`, `cfv_reconstruction_np8` |

[test_Reconstruction.cpp](../../test/cpp/CFV/test_Reconstruction.cpp)

- VR setup uses the small-mode cutoff and reduces owned-cell diagnostics
- Mesh dependency stamps reject stale reconstruction and permit explicit rebuild
- Granular metric publication requires current prerequisites and dependent rebuilds
- Native reconstruction matrix export preserves mesh state and exact face slots
- Finite-volume scoped query views retain native topology and reject stale publications
- Wall-distance replacement invalidates only reconstruction that depends on it
- CFV native authority preserves metrics and anisotropic reconstruction
- Wall/
- Periodic reconstruction convergence series
- VFV P2 HQM converges on IV10 base mesh
- Native periodic face slots match independent Quad4 vertex and neighbor oracle
- Limiter procedure: reconstruction + smooth indicator + WBAP limiter
- A-weighted limited variational reconstruction update has exact alpha endpoints and blend
- A-weighted limited variational reconstruction preserves GS and relaxation semantics
- Gradient conversion populates only O2 reconstruction modes
- O2 WBAP traversal agrees with full-order traversal on a linear target
- Limited Jacobi converges to the directly assembled penalized system
- Static reconstruction mask passed as a call argument constrains updates

## `dnds_test_amgx_preconditioner`

| Level | CTest registrations |
|---|---|
| 2 | `dnds_amgx_preconditioner` |

[test_AmgXPreconditioner.cpp](../../test/cpp/DNDS/test_AmgXPreconditioner.cpp)

- Native flow-five AMG preserves an uncoupled zero transverse equation
- Single-device AmgX applies a fixed linear local preconditioner and refreshes values
- Native flow and modal block inversion covers all launched thread scratch and dense entries
- Native matrix attachment shares values with fixed-cycle upload equivalence and refresh
- ACM block Jacobi preserves a null component through every geometric level and numeric refresh
- ACM block Jacobi converges for the finite SUBOFF diagonal after 49 refreshes

## `dnds_test_cauchy_spatial_schedule_cuda`

| Level | CTest registrations |
|---|---|
| 2 | `dnds_cauchy_spatial_schedule_cuda`, `dnds_cauchy_spatial_schedule_cuda_np2` |

[test_SpatialScheduleCUDA.cu](../../test/cpp/Cauchy/test_SpatialScheduleCUDA.cu)

- Flow AmgX matches periodic images and sums faces in the native CSR slots
- Prepared interface packet maps components and rotates vectors on every backend
- Masked mean validation skips unused interface rows without downloading state
- Boundary packet publication and frozen snapshots retain device placement
- An empty CUDA boundary publication stays current after the stream retires
- Boundary momentum reduction selects compact face ranges across inline worker and CUDA
- Spatial residual uses one managed schedule for inline worker and CUDA
- Managed nonlinear arithmetic retains CPU worker and CUDA state semantics
- Explicit stage algebra preserves conservative flow and independent RANS repair
- Visualization field preparation is resident and matches independent perfect-gas values
- Cauchy AMG reconstruction matches explicit CFV composition and omits sweep matrices

## `dnds_test_cauchy_time_fas_cuda`

| Level | CTest registrations |
|---|---|
| 2 | `dnds_cauchy_time_fas_cuda` |

[test_TimeFASCUDA.cu](../../test/cpp/Cauchy/test_TimeFASCUDA.cu)

- Resident equation checkpoint copies preserve padded layout and distributed row remapping
- Method-owned stage history uses instance-bound algebra on every execution backend
- Selected FAS algebra and cross-space transfer preserve frozen components

## `geom_test_mesh_index_conversion`

| Level | CTest registrations |
|---|---|
| 2 | `geom_mesh_index_conversion_np1`, `geom_mesh_index_conversion_np2`, `geom_mesh_index_conversion_np4`, `geom_mesh_index_conversion_np8` |

[test_MeshIndexConversion.cpp](../../test/cpp/Geom/test_MeshIndexConversion.cpp)

- Mesh setup: every rank has cells and nodes
- Mesh setup: multi-rank produces ghosts
- Mesh setup: periodic state matches configuration
- NodeIndex: Global2Local round-trip on father nodes
- NodeIndex: Global2Local round-trip on ghost nodes
- NodeIndex: _NoSon round-trip on father nodes
- NodeIndex: _NoSon returns negative for non-local global
- NodeIndex: Global2Local returns negative for unknown global
- CellIndex: Global2Local round-trip on father cells
- CellIndex: Global2Local round-trip on ghost cells
- CellIndex: _NoSon round-trip on father cells
- CellIndex: _NoSon returns negative for non-local global
- BndIndex: Global2Local round-trip on father bnds
- BndIndex: _NoSon round-trip on father bnds
- UnInitIndex pass-through for all 12 conversion methods
- Local2Global: negative local index decodes via -1-x encoding
- AdjPrimary: cell2node local indices are in valid range
- AdjPrimary: cell2cell local entries are valid or not-found
- AdjPrimary: bnd2cell owner cell is a local father cell
- AdjPrimary: bnd2node local indices are in valid range
- AdjPrimary: Local2Global then Global2Local is identity
- AdjPrimary: three consecutive round-trips are stable
- AdjPrimaryForBnd: detached cell2node conversion leaves all other published relations unchanged

## `geom_test_mesh_pipeline`

| Level | CTest registrations |
|---|---|
| 2 | `geom_mesh_pipeline_np1`, `geom_mesh_pipeline_np2`, `geom_mesh_pipeline_np4`, `geom_mesh_pipeline_np8` |

[test_MeshPipeline.cpp](../../test/cpp/Geom/test_MeshPipeline.cpp)

- Pipeline: all five published relation groups are Local
- Pipeline: global cell count matches expected
- Pipeline: global bnd count matches expected
- InterpolateFace: face count is positive
- InterpolateFace: face2cell has exactly 2 entries per face
- InterpolateFace: face2cell owner is a valid local cell
- InterpolateFace: face2node indices in valid range
- InterpolateFace: cell2face row sizes match expected face count
- InterpolateFace: cell2face entries are valid face indices
- InterpolateFace: bnd2face maps to valid faces
- InterpolateFace: face element types are valid
- InterpolateFace: independent element slots and exact back-face cones agree
- Periodic: face pbi has uniform XOR between owner and neighbor (collaborating check)
- Periodic: face2nodePbi is consistent with owner cell's cell2nodePbi
- Periodic: native output image count matches independent node-frame keys
- Periodic: ordered incidence PBI and physical geometry survive native reassembly
- bnd2cell: independent boundary facet incidence on all mesh configs
- BuildGhostPrimary: sparse closure and all global relations survive native reassembly
- Native inverse compose and face topology match independent incidences
- N2CB: every owned node has at least one adjacent cell
- N2CB: owned node stars match independent local cone incidence
- Cell face neighbors: row sizes match cell2face
- Cell face neighbors: each slot names the opposite face cell or boundary sentinel
- ConstructBndMesh: correct dimensions and native local state
- ConstructBndMesh: node2parentNode valid
- Boundary adjacency round-trip: native detached local and global states
- Ball2: mixed 3D element types
- Native output connectivity: arrays populated correctly
- Native output connectivity: repeat build replaces buffers without accumulating entries
- ReorderLocalCells: all published relation groups remain local
- ReorderLocalCells: cell count preserved
- ReorderLocalCells: partition starts valid
- ReorderLocalCells: cell2node still valid
- ReorderLocalCells: face count preserved
- AdjFacial detached round-trip: face2node and face2cell identity
- AdjC2F detached round-trip: cell2face and bnd2face identity
- AdjN2CB detached round-trip: node2cell and node2bnd identity
- AdjC2CFace detached round-trip: ordered face-neighbor identity
- AdjPrimary detached round-trip and serial output preserve source publication
- Elevation: O2 mesh cell count equals O1 cell count
- Elevation: O2 mesh has more nodes than O1
- Bisection: O1 mesh has more cells than O2
- Bisection: global node count is preserved from O2 mesh
- Elevation+Bisection: exact cell count progression
- BoundarySmooth: NACA0012 moves boundary O2 nodes
- BoundarySmooth: no NaN in displacement field
- InternalSmooth V2: coordinates remain finite after solve
- Elevation: O2 coordinates have no NaN
- MultiLayerGhost: ghost counts increase with layer count
- MultiLayerGhost: 2-layer cell2cell neighbors are all local
- MultiLayerGhost: global cell count preserved across layers
- InterpolateEdge: edge count is positive
- InterpolateEdge: edge2node indices in valid range
- InterpolateEdge: edge2cell has independently expected parents per edge
- InterpolateEdge: cell2edge row sizes match expected edge count
- InterpolateEdge: cell2edge entries have exact element-slot endpoint IDs
- InterpolateEdge: edge adjacency state is Local after pipeline
- InterpolateEdge: edge elem types are valid Line2 or Line3

## `geom_test_sycl_migration`

| Level | CTest registrations |
|---|---|
| 2 | `geom_sycl_cpu_migration_np2`, `geom_sycl_cuda_migration_np2` |

[test_SYCLMigration.cpp](../../test/cpp/Geom/test_SYCLMigration.cpp)

- SYCL queue and allocation owners cover table, strided and ragged rows
- SYCL mesh placement preserves owned and ghost incidence and coordinates
