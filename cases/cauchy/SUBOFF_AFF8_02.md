# SUBOFF AFF8-02 steady ACM–SST continuation

Use the corrected three-dimensional `data/mesh/SUBOFF_AFF8-02.cgns`, with farfield zone 2 and no-slip hull, sail and horizontal/vertical fins in zones 3–6. Both inputs assume a mesh in metres, scaled by 1/4.356 for solver coordinates. The flow has zero incidence, reference density 1000 kg/m³, speed 2.75482094 m/s and hull-length Reynolds number 12 million. The low-turbulence freestream in code units is k = 3.025e−10 and omega = 0.726. The sixth state component is omega in unit-code-density ACM. Use the same mesh and persistent cell IDs throughout the restart chain.

| Phase | File | Reconstruction and transport |
|---|---|---|
| Startup | [P1 Green–Gauss](acm_suboff_aff8_02_p1.json) | Direct gradients, mean RANS face values, CFL ramps to 20. |
| Continuation | [P2/Q3 VR](acm_suboff_aff8_02_p2q3.json) | One Jacobi VR sweep, full inverse and raw face matrix in FP32 storage with FP64 arithmetic. Replace k and omega modes by Barth-limited P1 gradients. Use reconstructed RANS face values and limited viscous reconstruction, RANS update relaxation 0.5 and CFL cap 10. |

Both use Roe and three fixed single-device AmgX defect corrections per nonlinear update. Output is per 2000 nonlinear updates for checkpoints and per 4000 for volume and boundary visualization. Use one GPU rank, 16 host threads and MPI without CPU binding. Run from the directory that contains `data/mesh/`; point `case_dir` and `solver` to the cases and executable from this checkout. Each invocation needs a fresh output prefix if previous results are to be preserved.

```bash
export OMP_NUM_THREADS=16 DNDS_MT_OMP_NUM_THREADS=16 DNDS_DIST_OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
case_dir=/path/to/DNDSR-Cauchy-ACM/cases/cauchy
solver=/path/to/DNDSR-Cauchy-ACM/build-acm-cuda/app/cauchy.exe
out=data/outUnsteady/Cauchy/SUBOFF_AFF8_02
mpirun --bind-to none -np 1 "$solver" "$case_dir/acm_suboff_aff8_02_p1.json"

# Reproduce the intermediate P2 stage from the P1 update-4000 means.
mpirun --bind-to none -np 1 "$solver" "$case_dir/acm_suboff_aff8_02_p2q3.json" \
  --restart "$out/P1/run-D0-S1_0_4000.h5" --restart-mode meansOnly \
  -k /domains/0/settings/numerics/ransCellCenterViscosity -v true \
  -k /domains/0/settings/numerics/cellCenterWallDistanceForQuadrature -v true \
  -k /output/prefix -v '"data/outUnsteady/Cauchy/SUBOFF_AFF8_02/P2-center/run"'

# Reproduce the latest P2 stage from the intermediate P2 update-4000 means.
mpirun --bind-to none -np 1 "$solver" "$case_dir/acm_suboff_aff8_02_p2q3.json" \
  --restart "$out/P2-center/run-D0-S1_0_4000.h5" --restart-mode meansOnly
```

The intermediate stage used averaged cell-center turbulent viscosity at faces and cell-center distances at quadrature. The latest stage disables both approximations and retains half the maximum nodal wall distance for the first-layer treatment. Means-only reads cold-start VR coefficients and nonlinear history, so update numbers are local to each invocation.

On this mesh the unbounded P2/Q3 trial and the unlimited RANS P1 overlay both developed an omega excursion and failed within 320 updates. With the Barth-limited RANS P1 overlay the matched center-viscosity run passed 5000 updates; the subsequent face-viscosity run also passed 4000. These are stable continuation checkpoints. The selected velocity residual remained above its configured steady convergence threshold.
