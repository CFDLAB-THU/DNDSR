# Historical validation and correction records

These records describe the revisions and cases evaluated at the time. For the current operator contracts, see the [architecture index](../sphinx/architecture.md).

The [2026-10-31 VR SVD correction](vr_svd_cutoff_2026-10-31.md) documents the
small-mode cutoff change, setup diagnostics and affected nonzero-tolerance cases.

The [2026-11-03 extended Euler rotation correction](euler_extended_rotation_2026-11-03.md)
records the NS_EX/NS_EX_3D absolute-velocity source convention and focused tests.

The [Cauchy preliminary completion record](cauchy_preliminary_2026-11-01.md)
records physical layouts, resident CUDA reconstruction, explicit device-packed
transfers, completed gates and performance limits.

The [NCCL/FP64 complement record](nccl_fp64_2026-11-01.md) documents the
CUDA-default NCCL build option, explicit provider/kernel selection, final
distributed gates and the measured gains and regressions by shape.

The [Plan 61-6 consolidation record](plan616_cuda_consolidation.md) describes
the bounded generated RHS/VR catalogs, shared helpers and native AMG exploration.

The [Plan 61 CUDA inventory](plan61_cuda_inventory.md) tracks the specialized
perfect-gas device path and its grouped implementation/validation status.

The [November 5 pMG and distributed-failure audit](cauchy_pmg_distributed_errors_2026-11-05.md)
records the coarse-gradient correction, fail-fast boundary and publication checks.

The [Cauchy audit fix record](cauchy_audit_fix.md) tracks the shared SST
blend correction and independent RANS reconstruction positivity policy.
The [plan 53 execution ledger](plan53_audit_execution.md) reconciles those
commits with the remaining programming audit, UT consolidation and enhancements.

The mesh construction correction and its fixture/condition audit are recorded
in the validation ledger below. Earlier infrastructure acceptance was withdrawn
after detecting an omitted boundary ring; see the
[evidence correction](infrastructure_finish_2026-10-27.md). Test totals and
owned numerical parity do not prove per-rank ghost completeness.
The [Runtime/mesh audit](runtime_mesh_audit_2026-10-29.md) records the
abstraction and safety review, fixes, validation checkpoints and the completed
parallel setup median target; serial import and later FV work are separate.
The [2026-10-30 retrospective](refactor_retrospective_2026-10-30.md) records
the next audit's local fixes, remaining design decisions, test limitations and
CFV/sliding-interface planning boundary. Earlier validation is not evidence for
these new changes.
The [accepted-fixes follow-up](accepted_audit_fixes_2026-10-30.md) resolves
the field-layout and compiled-program decisions and records the next FV/Geom
work orders.
The [CPU FV/interface validation](fv_interface_2026-10-30.md) records the
implemented numerical/geometric scope, focused tests and matched legacy VR
performance evidence.
The [2026-10-31 facility next steps](facilities_next_steps_2026-10-31.md)
separates current DNDS/Geom/CFV support from accepted overlap, numerical and
distributed-interface work, including explicit deferred features.
The [source comparison and validation ledger](mesh_construction_semantics_2026-10-28.md)
records the final checks, explicit legacy periodic-frame differences and limits.
The [legacy connectivity test-purpose ledger](mesh_connectivity_test_port_2026-10-28.md)
records all 47 original cases, independent expected answers and explicit
replacements for retired containers.

The [2026-11-05 Euler gradient correction](euler_gradient_buffers_2026-11-05.md)
documents limited-gradient preparation, explicit reuse and boundary-stage timing.

The [Euler RANS reconstruction PP correction](euler_rans_reconstruction_pp_2026-11-05.md)
separates spatial flow beta from scalar limiting and preserves coupled increment scaling.

The [SST convergence audit](euler_sst_convergence_2026-11-05.md)
records source linearization, wall perturbation checks and O2 numerical trials.

The [GG boundary-target correction](euler_gg_boundary_targets_2026-11-05.md)
documents mutable wall traces, immutable means, preserved VR semantics and
EulerP scalar neighbour/boundary indexing.
